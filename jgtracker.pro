TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

# When a memory error occurs, use Address sanitizer:
# QMAKE_CXXFLAGS_DEBUG += -fsanitize=address -fno-omit-frame-pointer
# QMAKE_LFLAGS_DEBUG += -fsanitize=address

# export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer-3.5
# export ASAN_OPTIONS=symbolize=1
# ./jgtracker params

INCLUDEPATH += src src/jgtracker/thirdparty/

HEADERS += \
    src/jgtracker/thirdparty/cppmt/fastcluster/fastcluster.h \
    src/jgtracker/thirdparty/cppmt/logging/log.h \
    src/jgtracker/thirdparty/cppmt/CMT.h \
    src/jgtracker/thirdparty/cppmt/Consensus.h \
    src/jgtracker/thirdparty/cppmt/Fusion.h \
    src/jgtracker/thirdparty/cppmt/gui.h \
    src/jgtracker/thirdparty/cppmt/Matcher.h \
    src/jgtracker/thirdparty/cppmt/Tracker.h \
    src/jgtracker/thirdparty/mt/assert.h \
    src/jgtracker/thirdparty/mt/check.h \
    src/jgtracker/thirdparty/mt/common.h \
    src/jgtracker/thirdparty/mt/fileio.h \
    src/jgtracker/thirdparty/mt/memory.h \
    src/jgtracker/thirdparty/mt/unicode.h \
    src/jgtracker/thirdparty/mt/varint.h \
    src/jgtracker/Evaluator.h \
    src/jgtracker/FeaturesExtractor.h \
    src/jgtracker/Histogram.h \
    src/jgtracker/Manager.h \
    src/jgtracker/MatchFinder.h \
    src/jgtracker/SIRParticleFilter.h \
    src/jgtracker/Target.h \
    src/jgtracker/TargetCreator.h \
    src/jgtracker/Voting.h \
    src/jgtracker/HistogramFactory.h \
    src/jgtracker/types.h \
    src/jgtracker/operations.h \
    src/jgtracker/TargetSelector.h \
    src/jgtracker/IntegralHistogramFactory.h \
    src/jgtracker/thirdparty/cppmt/utils.h


SOURCES += \
    src/jgtracker/thirdparty/cppmt/fastcluster/fastcluster.cpp \
    src/jgtracker/thirdparty/cppmt/CMT.cpp \
    src/jgtracker/thirdparty/cppmt/Consensus.cpp \
    src/jgtracker/thirdparty/cppmt/Fusion.cpp \
    src/jgtracker/thirdparty/cppmt/gui.cpp \
    src/jgtracker/thirdparty/cppmt/Matcher.cpp \
    src/jgtracker/thirdparty/cppmt/Tracker.cpp \
    src/jgtracker/thirdparty/mt/assert.cpp \
    src/jgtracker/thirdparty/mt/check.cpp \
    src/jgtracker/thirdparty/mt/common.cpp \
    src/jgtracker/thirdparty/mt/fileio.cpp \
    src/jgtracker/thirdparty/mt/memory.cpp \
    src/jgtracker/thirdparty/mt/unicode.cpp \
    src/jgtracker/thirdparty/mt/varint.cpp \
    src/jgtracker/Evaluator.cpp \
    src/jgtracker/FeaturesExtractor.cpp \
    src/jgtracker/Histogram.cpp \
    src/jgtracker/main.cpp \
    src/jgtracker/Manager.cpp \
    src/jgtracker/MatchFinder.cpp \
    src/jgtracker/SIRParticleFilter.cpp \
    src/jgtracker/Target.cpp \
    src/jgtracker/TargetCreator.cpp \
    src/jgtracker/TargetSelector.cpp \
    src/jgtracker/Voting.cpp \
    src/jgtracker/HistogramFactory.cpp \
    src/jgtracker/types.cpp \
    src/jgtracker/operations.cpp \
    src/jgtracker/IntegralHistogramFactory.cpp \
    src/jgtracker/thirdparty/cppmt/utils.cpp

TARGET = jgtracker

LIBS += -lboost_filesystem -lboost_iostreams -lboost_system
LIBS += `pkg-config --libs opencv`
