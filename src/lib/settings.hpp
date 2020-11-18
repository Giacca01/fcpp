// Copyright © 2019 Giorgio Audrito. All Rights Reserved.

/**
 * @file settings.hpp
 * @brief Definition of default values for settings macros.
 */

#ifndef FCPP_SETTINGS_H_
#define FCPP_SETTINGS_H_

#include <cstdint>
#include <limits>


//! @brief Identifier for low-end, resource constrained systems.
#define FCPP_SYSTEM_EMBEDDED 11
//! @brief Identifier for high-end, general purpose systems.
#define FCPP_SYSTEM_GENERAL  22

#ifndef FCPP_SYSTEM
//! @brief Setting defining the system: @ref FCPP_SYSTEM_EMBEDDED or @ref FCPP_SYSTEM_GENERAL (default).
#define FCPP_SYSTEM FCPP_SYSTEM_GENERAL
#endif


//! @brief Identifier for logical cloud systems, not simulating a physical world.
#define FCPP_ENVIRONMENT_LOGICAL   111
//! @brief Identifier for physically deployed systems.
#define FCPP_ENVIRONMENT_PHYSICAL  222
//! @brief Identifier for simulations of deployed systems.
#define FCPP_ENVIRONMENT_SIMULATED 333


#ifndef FCPP_ENVIRONMENT
//! @brief Setting defining the overall environment: @ref FCPP_ENVIRONMENT_LOGICAL, @ref FCPP_ENVIRONMENT_PHYSICAL or @ref FCPP_ENVIRONMENT_SIMULATED (default).
#define FCPP_ENVIRONMENT FCPP_ENVIRONMENT_SIMULATED
#endif


//! @brief Identifier for systems operating stand-alone, without user or network interactions.
#define FCPP_CONFIGURATION_STANDALONE 1111
//! @brief Identifier for systems depending on user or network interaction.
#define FCPP_CONFIGURATION_DEPENDENT  2222

#ifndef FCPP_CONFIGURATION
//! @brief Setting defining the overall environment: @ref FCPP_CONFIGURATION_STANDALONE (default) or @ref FCPP_CONFIGURATION_DEPENDENT.
#define FCPP_CONFIGURATION FCPP_CONFIGURATION_STANDALONE
#endif


#if   FCPP_SYSTEM == FCPP_SYSTEM_GENERAL
    #ifndef FCPP_TRACE
    //! @brief Setting defining the size of trace hashes (64 for general systems, 16 for embedded systems).
    #define FCPP_TRACE 64
    #endif
    #ifndef FCPP_DEVICE
    //! @brief Setting defining the size of device identifiers (32 for general systems, 16 for embedded systems).
    #define FCPP_DEVICE 32
    #endif
    #ifndef FCPP_HOPS
    //! @brief Setting defining the size of hop counts (16 for general systems, 8 for embedded systems).
    #define FCPP_HOPS 16
    #endif
#elif FCPP_SYSTEM == FCPP_SYSTEM_EMBEDDED
    #ifndef FCPP_TRACE
    //! @brief Setting defining the size of trace hashes (64 for general systems, 16 for embedded systems).
    #define FCPP_TRACE 16
    #endif
    #ifndef FCPP_DEVICE
    //! @brief Setting defining the size of device identifiers (32 for general systems, 16 for embedded systems).
    #define FCPP_DEVICE 16
    #endif
    #ifndef FCPP_HOPS
    //! @brief Setting defining the size of hop counts (16 for general systems, 8 for embedded systems).
    #define FCPP_HOPS 8
    #endif
#else
    static_assert(false, "invalid value for FCPP_SYSTEM");
#endif


#if   FCPP_ENVIRONMENT == FCPP_ENVIRONMENT_LOGICAL
    #ifndef FCPP_EXPORT_NUM
    //! @brief Setting defining whether exports for self and other devices should be separated (2, default for physical systems) or together (1, default for simulated and logical systems).
    #define FCPP_EXPORT_NUM 1
    #endif
    #ifndef FCPP_EXPORT_PTR
    //! @brief Setting defining whether exports should be handled as values (false, default for physical systems) or shared pointers (true, default for simulated and logical systems).
    #define FCPP_EXPORT_PTR true
    #endif
    #ifndef FCPP_ONLINE_DROP
    //! @brief Setting defining whether old messages should be dropped while new ones arrive (true, default for physical systems) or at round start (false, default for simulated and logical systems).
    #define FCPP_ONLINE_DROP false
    #endif
    #ifndef FCPP_PARALLEL
    //! @brief Setting defining whether the computation should be performed with parallel threads.
    #define FCPP_PARALLEL true
    #endif
    #ifndef FCPP_SYNCHRONISED
    //! @brief Setting defining whether many events are expected to happen at the same time.
    #define FCPP_SYNCHRONISED true
    #endif
#elif FCPP_ENVIRONMENT == FCPP_ENVIRONMENT_PHYSICAL
    #ifndef FCPP_EXPORT_NUM
    //! @brief Setting defining whether exports for self and other devices should be separated (2, default for physical systems) or together (1, default for simulated and logical systems).
    #define FCPP_EXPORT_NUM 2
    #endif
    #ifndef FCPP_EXPORT_PTR
    //! @brief Setting defining whether exports should be handled as values (false, default for physical systems) or shared pointers (true, default for simulated and logical systems).
    #define FCPP_EXPORT_PTR false
    #endif
    #ifndef FCPP_ONLINE_DROP
    //! @brief Setting defining whether old messages should be dropped while new ones arrive (true, default for physical systems) or at round start (false, default for simulated and logical systems).
    #define FCPP_ONLINE_DROP true
    #endif
    #ifndef FCPP_PARALLEL
    //! @brief Setting defining whether the computation should be performed with parallel threads.
    #define FCPP_PARALLEL false
    #endif
    #ifndef FCPP_SYNCHRONISED
    //! @brief Setting defining whether many events are expected to happen at the same time.
    #define FCPP_SYNCHRONISED false
    #endif
#elif FCPP_ENVIRONMENT == FCPP_ENVIRONMENT_SIMULATED
    #ifndef FCPP_EXPORT_NUM
    //! @brief Setting defining whether exports for self and other devices should be separated (2, default for physical systems) or together (1, default for simulated and logical systems).
    #define FCPP_EXPORT_NUM 1
    #endif
    #ifndef FCPP_EXPORT_PTR
    //! @brief Setting defining whether exports should be handled as values (false, default for physical systems) or shared pointers (true, default for simulated and logical systems).
    #define FCPP_EXPORT_PTR true
    #endif
    #ifndef FCPP_ONLINE_DROP
    //! @brief Setting defining whether old messages should be dropped while new ones arrive (true, default for physical systems) or at round start (false, default for simulated and logical systems).
    #define FCPP_ONLINE_DROP false
    #endif
    #ifndef FCPP_PARALLEL
    //! @brief Setting defining whether the computation should be performed with parallel threads.
    #define FCPP_PARALLEL false
    #endif
    #ifndef FCPP_SYNCHRONISED
    //! @brief Setting defining whether many events are expected to happen at the same time.
    #define FCPP_SYNCHRONISED false
    #endif
#else
    static_assert(false, "invalid value for FCPP_ENVIRONMENT");
#endif


#ifndef FCPP_THREADS
    #if FCPP_PARALLEL == true
    //! @brief Setting regulating the number of threads to be used.
    #define FCPP_THREADS std::thread::hardware_concurrency()
    #else
    //! @brief Setting regulating the number of threads to be used.
    #define FCPP_THREADS 1
    #endif
#endif


#ifndef FCPP_REALTIME
    #if FCPP_ENVIRONMENT == FCPP_ENVIRONMENT_PHYSICAL || FCPP_CONFIGURATION == FCPP_CONFIGURATION_DEPENDENT
    //! @brief Factor multiplying real time passing (1 for physical or dependent systems, infinity for others).
    #define FCPP_REALTIME 1.0
    #else
    //! @brief Factor multiplying real time passing (1 for physical or dependent systems, infinity for others).
    #define FCPP_REALTIME std::numeric_limits<double>::infinity()
    #endif
#endif


#ifndef FCPP_MESSAGE_PUSH
    //! @brief Setting defining whether incoming messages are pushed or pulled.
    #define FCPP_MESSAGE_PUSH true
#endif


#ifndef FCPP_VALUE_PUSH
    //! @brief Setting defining whether new values should be pushed to aggregators or pulled when needed.
    #define FCPP_VALUE_PUSH false
#endif


#ifndef FCPP_WARNING_TRACE
    //! @brief Setting defining whether hash colliding of code points is admissible.
    #define FCPP_WARNING_TRACE true
#endif


#ifndef FCPP_TIME_TYPE
//! @brief Setting defining the type to be used to represent times (double for simulations, time_t for physical and logical systems).
#define FCPP_TIME_TYPE double
#endif
#ifndef FCPP_TIME_EPSILON
//! @brief Setting defining which time differences are to be considered negligible.
#define FCPP_TIME_EPSILON 0.01
#endif


/**
 * @brief Namespace containing all the objects in the FCPP library.
 */
namespace fcpp {
    //! @brief Type used for times.
    using times_t = FCPP_TIME_TYPE;
    //! @brief Minimum time (infinitely in the past).
    constexpr times_t TIME_MIN = std::numeric_limits<times_t>::has_infinity ? -std::numeric_limits<times_t>::infinity() : std::numeric_limits<times_t>::lowest();
    //! @brief Maximum time (infinitely in the future).
    constexpr times_t TIME_MAX = std::numeric_limits<times_t>::has_infinity ? std::numeric_limits<times_t>::infinity() : std::numeric_limits<times_t>::max();
    //! @brief Shorthand to double infinity value.
    constexpr double INF = std::numeric_limits<double>::infinity();

#if   FCPP_DEVICE == 8
    typedef uint8_t device_t;
#elif FCPP_DEVICE == 16
    typedef uint16_t device_t;
#elif FCPP_DEVICE == 24
    typedef uint32_t device_t;
#elif FCPP_DEVICE == 32
    typedef uint32_t device_t;
#elif FCPP_DEVICE == 48
    typedef uint64_t device_t;
#elif FCPP_DEVICE == 64
    typedef uint64_t device_t;
#else
    static_assert(false, "invalid value for FCPP_DEVICE");
    //! @brief Type for device identifiers (depends on @ref FCPP_DEVICE).
    typedef uint64_t device_t;
#endif

#if   FCPP_HOPS == 8
    typedef int8_t hops_t;
#elif FCPP_HOPS == 16
    typedef int16_t hops_t;
#elif FCPP_HOPS == 32
    typedef int32_t hops_t;
#else
    static_assert(false, "invalid value for FCPP_HOPS");
    //! @brief Type for hop counts (depends on @ref FCPP_HOPS).
    typedef int32_t hops_t;
#endif
}


#endif // FCPP_SETTINGS_H_
