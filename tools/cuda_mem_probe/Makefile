# cuda_mem_probe — build the LD_PRELOAD shared library.
# No CUDA headers/libs needed: wrappers are typed with void*/size_t and resolve
# the real symbols at runtime via dlsym. Just a C compiler + dlfcn + pthread.
CC      ?= cc
CFLAGS  ?= -O2 -fPIC -Wall -Wextra
LDFLAGS ?= -shared -ldl -lpthread

TARGET  = libcudamemprobe.so

$(TARGET): cuda_mem_probe.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: test clean
test: $(TARGET)
	bash tests/run_selftest.sh

clean:
	rm -f $(TARGET) tests/probe_selftest
