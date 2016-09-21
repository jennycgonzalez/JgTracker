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

INCLUDEPATH += src

HEADERS += \
    src/jg/thirdparty/cppmt/fastcluster/fastcluster.h \
    src/jg/thirdparty/cppmt/logging/log.h \
    src/jg/thirdparty/cppmt/CMT.h \
    src/jg/thirdparty/cppmt/common.h \
    src/jg/thirdparty/cppmt/Consensus.h \
    src/jg/thirdparty/cppmt/Fusion.h \
    src/jg/thirdparty/cppmt/gui.h \
    src/jg/thirdparty/cppmt/Matcher.h \
    src/jg/thirdparty/cppmt/Tracker.h \
    src/jg/Evaluator.h \
    src/jg/FeaturesExtractor.h \
    src/jg/Histogram.h \
    src/jg/Manager.h \
    src/jg/MatchFinder.h \
    src/jg/SIRParticleFilter.h \
    src/jg/Target.h \
    src/jg/TargetCreator.h \
    src/jg/Voting.h \
    src/jg/HistogramFactory.h \
    src/jg/types.h \
    src/jg/operations.h \
    src/jg/TargetSelector.h \
    src/jg/IntegralHistogramFactory.h

SOURCES += \
    src/jg/thirdparty/cppmt/fastcluster/fastcluster.cpp \
    src/jg/thirdparty/cppmt/CMT.cpp \
    src/jg/thirdparty/cppmt/common.cpp \
    src/jg/thirdparty/cppmt/Consensus.cpp \
    src/jg/thirdparty/cppmt/Fusion.cpp \
    src/jg/thirdparty/cppmt/gui.cpp \
    src/jg/thirdparty/cppmt/Matcher.cpp \
    src/jg/thirdparty/cppmt/Tracker.cpp \
    src/jg/Evaluator.cpp \
    src/jg/FeaturesExtractor.cpp \
    src/jg/Histogram.cpp \
    src/jg/main.cpp \
    src/jg/Manager.cpp \
    src/jg/MatchFinder.cpp \
    src/jg/SIRParticleFilter.cpp \
    src/jg/Target.cpp \
    src/jg/TargetCreator.cpp \
    src/jg/TargetSelector.cpp \
    src/jg/Voting.cpp \
    src/jg/HistogramFactory.cpp \
    src/jg/types.cpp \
    src/jg/operations.cpp \
    src/jg/IntegralHistogramFactory.cpp

TARGET = jgtracker

LIBS += -lboost_filesystem -lboost_system -lmt
LIBS += `pkg-config --libs opencv`


#LIBS += -ltcmalloc_minimal
# Only needed when Google's PerfTools are used.
#LIBS += -lprofiler
