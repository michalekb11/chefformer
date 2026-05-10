# UPDATE: To make use of Apple MPS, avoid using docker and use local python env directly.
#build-training-image:
# 	docker build \
# 	-f docker/training/train_model \
# 	-t chefformer-train:latest \
# 	.