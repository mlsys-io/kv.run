#!/bin/bash

# Kubernetes deployment script for MLOC

set -e

# Configuration
NAMESPACE="mloc"
REGISTRY="your-registry.com"
IMAGE_TAG="latest"

echo "🚀 Deploying MLOC to Kubernetes..."

# Create namespace if it doesn't exist
echo "📦 Creating namespace: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply Redis deployment
echo "📦 Deploying Redis..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: ["redis-server", "--appendonly", "yes"]
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

# Apply Orchestrator deployment
echo "📦 Deploying Orchestrator..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: $REGISTRY/mloc:$IMAGE_TAG
        ports:
        - containerPort: 8000
        env:
        - name: MLOC_NODE_TYPE
          value: "ORCHESTRATOR"
        - name: MLOC_REDIS_URL
          value: "redis://redis:6379"
        - name: MLOC_HOST
          value: "0.0.0.0"
        - name: MLOC_PORT
          value: "8000"
        - name: MLOC_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator
  namespace: $NAMESPACE
spec:
  selector:
    app: orchestrator
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
EOF

# Apply Worker deployment
echo "📦 Deploying Workers..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker
  namespace: $NAMESPACE
spec:
  replicas: 3  # Adjust based on your needs
  selector:
    matchLabels:
      app: worker
  template:
    metadata:
      labels:
        app: worker
    spec:
      containers:
      - name: worker
        image: $REGISTRY/mloc:$IMAGE_TAG
        env:
        - name: MLOC_NODE_TYPE
          value: "WORKER"
        - name: MLOC_REDIS_URL
          value: "redis://redis:6379"
        - name: MLOC_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: worker-data
          mountPath: /tmp/mloc_work
        - name: shared-models
          mountPath: /shared/models
      volumes:
      - name: worker-data
        emptyDir: {}
      - name: shared-models
        persistentVolumeClaim:
          claimName: shared-models-pvc
      nodeSelector:
        accelerator: nvidia-tesla-gpu  # Adjust based on your cluster
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-models-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
EOF

# Apply RBAC (if needed)
echo "📦 Applying RBAC..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mloc
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: mloc-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: mloc-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: mloc-role
subjects:
- kind: ServiceAccount
  name: mloc
  namespace: $NAMESPACE
EOF

echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/orchestrator -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/worker -n $NAMESPACE

echo "✅ MLOC deployed successfully!"
echo ""
echo "🔗 Access the Orchestrator API:"
EXTERNAL_IP=$(kubectl get service orchestrator -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$EXTERNAL_IP" ]; then
    echo "   http://$EXTERNAL_IP:8000"
else
    echo "   kubectl port-forward service/orchestrator 8000:8000 -n $NAMESPACE"
fi
echo ""
echo "📊 Monitor the cluster:"
echo "   kubectl get pods -n $NAMESPACE"
echo "   kubectl logs -f deployment/orchestrator -n $NAMESPACE"
echo "   kubectl logs -f deployment/worker -n $NAMESPACE"