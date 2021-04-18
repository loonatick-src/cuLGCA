CC = nvcc
CFLAGS = -g -Iinclude

all:
	$(CC) $(CFLAGS) src/init.cu -o bin/init.out

