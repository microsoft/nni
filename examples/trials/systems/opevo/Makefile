rocm-env: build
	docker run -it --rm --privileged -v /:/host -w /root \
		-e BACKEND=c-rocm -p 8080:8080 \
		tvm4nni bash || true

cuda-env: build
	docker run -it --rm --privileged -v /:/host -w /root \
		-e BACKEND=c-cuda -p 8080:8080 \
		-v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
		-v $(shell dirname `ldd /usr/lib/x86_64-linux-gnu/libcuda.so.1 | grep nvidia-fatbinaryloader | awk '{print $$3}'`):/usr/local/nvidia/lib64 \
		tvm4nni bash || true

build:
	docker build -t tvm4nni --network=host .
