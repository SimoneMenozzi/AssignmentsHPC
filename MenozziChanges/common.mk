INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).cu
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

DEPS        := Makefile.dep
DEP_FLAG    := -MM

# Compilatori
CC=gcc
NVCC=/usr/local/cuda/bin/nvcc # Path al compilatore CUDA
LD=ld
OBJDUMP=objdump

# Opzioni di ottimizzazione
OPT=-O2 -g
CFLAGS=$(OPT) -I. $(EXT_CFLAGS)
NVFLAGS=$(CFLAGS) -Xcompiler -fopenmp -lineinfo
LDFLAGS=-lm -lcudart $(EXT_LDFLAGS)

.PHONY: all exe clean veryclean run profile metrics

# Regola principale
all : exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(NVCC) $(NVFLAGS) $(INCPATHS) -x cu $^ -o $@ $(LDFLAGS)

clean :
	-rm -vf -vf $(EXE) *~ 

veryclean : clean
	-rm -vf $(DEPS)

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH} nvprof ./$(EXE)

metrics: $(EXE)
	sudo LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH} nvprof --print-gpu-trace --metrics "eligible_warps_per_cycle,achieved_occupancy,sm_efficiency,ipc" ./$(EXE)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)
