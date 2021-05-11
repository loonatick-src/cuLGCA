CC = nvcc
CFLAGS = -g -O2 -Wall -Wextra -Isrc -rdynamic -DNDEBUG $(OPTFLAGS)
LIBS = -ldl $(OPTLIBS)
PREFIX ?= /usr/local

SOURCES=$(wildcard src/**/*.cu src/*.cu)
OBJECTS=$(patsubst %.cu, %.o, $(CUSOURCES))

TEST_SRC=$(wildcard tests/*_tests.cu)

TARGET=build/libfhp.a
SO_TARGET=$(patsubst %.a, %.so, $(TARGET))

all: $(TARGET) $(SO_TARGET) tests

dev: CFLAGS = -g -Wall -Isrc -Wall -Wextra $(OPTFLAGS)
dev: all

$(TARGET): CFLAGS += -fPIC
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

clean:
	rm -rf build $(OBJECTS) $(TESTS)
	rm -rf tests/*.o
	rm -f tests/tests.log
	find . -name "*.gc*" -exec rm {} \;
	rm -rf `find . -name "*.dSYM" -print`
