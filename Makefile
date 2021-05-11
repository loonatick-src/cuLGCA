CC = nvcc
CFLAGS = -g -O2  -Isrc -DNDEBUG $(OPTFLAGS)
LIBS = -ldl $(OPTLIBS)
PREFIX ?= /usr/local

SOURCES=$(wildcard src/**/*.cu src/*.cu)
OBJECTS=$(patsubst %.cu, %.o, $(SOURCES))

TEST_SRC=$(wildcard tests/*_tests.cu)
TESTS=$(patsubst %.cu, %, $(TEST_SRC))

TARGET=build/libfhp.a
SO_TARGET=$(patsubst %.a, %.so, $(TARGET))

all: $(TARGET) $(SO_TARGET) tests

dev: CFLAGS = -g  -Isrc $(OPTFLAGS)
dev: all

src/%.o: src/%.cu
	$(CC) -c $(CFLAGS) $< -o $@

$(TARGET): build $(OBJECTS)
	ar rcs $@ $(OBJECTS)
	ranlib $@

$(SO_TARGET): $(TARGET) $(OBJECTS)
	$(CC) -shared -o $@ $(OBJECTS)

build:
	@mkdir -p build
	@mkdir -p bin

.PHONY: tests
tests: LDLIBS = $(TARGET)
tests: $(TESTS)
	sh ./tests/runtests.sh

$(TESTS): $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -rf build $(OBJECTS) $(TESTS)
	rm -rf tests/*.o
	rm -f tests/tests.log
	find . -name "*.gc*" -exec rm {} \;
	rm -rf `find . -name "*.dSYM" -print`
