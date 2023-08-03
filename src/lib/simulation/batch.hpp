// Copyright © 2023 Giorgio Audrito. All Rights Reserved.

/**
 * @file batch.hpp
 * @brief Helper functions for running a batch of simulations.
 */

#ifndef FCPP_SIMULATION_BATCH_H_
#define FCPP_SIMULATION_BATCH_H_

#include <cassert>

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef FCPP_MPI
#include <mpi.h>
#endif

#include "lib/common/algorithm.hpp"
#include "lib/common/option.hpp"
#include "lib/common/tagged_tuple.hpp"
#include "lib/component/logger.hpp"


/**
 * @brief Namespace containing all the objects in the FCPP library.
 */
namespace fcpp {


//! @brief Namespace containing tools for batch execution of FCPP simulations.
namespace batch {


//! @brief Namespace of tags for batch runs.
namespace tags {
    //! @brief A tag for indexing network types to be run.
    struct type_index {};
} // tags


//! @cond INTERNAL
namespace details {
    //! @brief Class wrapping a generating function with output type and size information.
    template <typename F, typename... Ts>
    class generator {
      public:
        //! @brief The tuple type that the function generates.
        using value_type = common::tagged_tuple<typename common::type_sequence<Ts...>::template slice<0, sizeof...(Ts)/2>, typename common::type_sequence<Ts...>::template slice<sizeof...(Ts)/2, sizeof...(Ts)>>;

        //! @brief Constructor setting up the class members.
        generator(F&& f, size_t core_size, size_t extra_size) : m_core_size(core_size), m_extra_size(extra_size), m_function(std::move(f)) {}

        /**
         * @brief Operator calling the wrapped function.
         *
         * @param t The tuple in which to store the values generated by the function.
         * @param i The index of the element to be generated.
         * @return  A boolean telling whether the given index has to be included (true) or skipped (false).
         */
        template <typename T>
        inline bool operator()(T& t, size_t i) const {
            return m_function(t, i);
        }

        //! @brief Returns the size of the core sequence that has to be expanded with every other value.
        inline size_t core_size() const {
            return m_core_size;
        }

        //! @brief Returns the size of the extra sequence that should be expanded only with core values.
        inline size_t extra_size() const {
            return m_extra_size;
        }

      private:
        //! @brief The size of the core sequence that has to be expanded with every other value.
        const size_t m_core_size;
        //! @brief The size of the extra sequence that should be expanded only with core values.
        const size_t m_extra_size;
        //! @brief The wrapped function.
        const F m_function;
    };

    //! @brief Wraps a generating function with output type and size information.
    template <typename... Ts, typename F>
    inline generator<F, Ts...> make_generator(F&& f, size_t core_size, size_t extra_size) {
        return {std::move(f), core_size, extra_size};
    }
} // details
//! @endcond


//! @brief Functor generating a single tuple with several constants.
template <typename... Ss, typename... Ts>
auto constant(Ts const&... xs) {
    return details::make_generator<Ss..., Ts...>([=](auto& t, size_t){
        t = common::make_tagged_tuple<Ss...>(xs...);
        return true;
    }, 1, 0);
}

//! @brief Functor generating a sequence of given values of type `char const*`. [DEPRECATED]
template <typename S, typename... Ts>
auto literals(char const* s, Ts const&... xs) {
    static_assert(common::always_false<S>::value, "the batch::literals function has been deprecated and should not be used, use batch::list instead");
    return details::make_generator<S, std::string>([=](auto&, size_t){
        return false;
    }, 0, 0);
}

//! @brief Functor generating a sequence of given values (`char const*` are wrapped as `std::string`).
template<typename S, typename T, typename... Ts>
auto list(T&& x, Ts&&... xs) {
    using DT = std::decay_t<T>;
    using CT = std::conditional_t<std::is_same<DT, char const*>::value, std::string, DT>;
    std::array<CT, sizeof...(Ts)+1> values{x, xs...};
    return details::make_generator<S, CT>([=](auto& t, size_t i){
        common::get<S>(t) = values[i];
        return true;
    }, values.size(), 0);
}

/**
 * @brief Functor generating a sequence of given core and extra values.
 *
 * Core values are expanded with value produced by the other generators.
 * Extra values are expanded only with core values produced by other generators.
 */
template<typename S, typename T>
auto double_list(std::vector<T> core, std::vector<T> extra) {
    for (T const& x : extra) for (T const& y : core)
        assert(x != y && "error: arguments to double_list have non-empty intersection");
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = i < core.size() ? core[i] : extra[i - core.size()];
        return true;
    }, core.size(), extra.size());
}

//! @brief Functor generating an arithmetic sequence of values (inclusive range).
template <typename S, typename T>
auto arithmetic(T min, T max, T step) {
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = min + i * step;
        return true;
    }, (max-min)/step + 1, 0);
}

/**
 * @brief Functor generating an arithmetic sequence (inclusive) with a default value.
 *
 * Only the default value is used when other defaulted parameters are set to extra values.
 */
template <typename S, typename T>
auto arithmetic(T min, T max, T step, T def) {
    size_t id = (def - min) / step;
    bool defextra = def != (min + id * step);
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = i == 0 ? def : defextra or i <= id ? min + (i-1) * step : min + i * step;
        return true;
    }, 1, (max-min)/step + defextra);
}

/**
 * @brief Functor generating an arithmetic sequence (inclusive) with a default range and extra range.
 *
 * Only the range `defmin...defmax` is used when other defaulted parameters are set to extra values.
 * The range `min...max` is used only with core values from other defaulted parameters.
 * The two ranges are assumed not to overlap.
 */
template <typename S, typename T>
auto arithmetic(T min, T max, T step, T defmin, T defmax) {
    size_t core_size = (defmax-defmin)/step + 1;
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = i < core_size ? defmin + i*step : min + (i-core_size) * step;
        return true;
    }, core_size, (max-min)/step + 1);
}

//! @brief Functor generating a geometric sequence of values (inclusive range).
template <typename S, typename T>
auto geometric(T min, T max, T step) {
    std::vector<T> v = {min};
    while (v.back() * step <= max) v.push_back(v.back() * step);
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = v[i];
        return true;
    }, v.size(), 0);
}

/**
 * @brief Functor generating a geometric sequence (inclusive) with a default value.
 *
 * Only the default value is used when other defaulted parameters are set to extra values.
 */
template <typename S, typename T>
auto geometric(T min, T max, T step, T def) {
    std::vector<T> v = {def};
    if (def != min) v.push_back(min);
    while (v.back() * step <= max) {
        if (v.size() > 1 and v.back() == def) v.back() *= step;
        else v.push_back(v.back() * step);
    }
    if (v.size() > 1 and v.back() == def) v.pop_back();
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = v[i];
        return true;
    }, 1, v.size() - 1);
}

/**
 * @brief Functor generating an geometric sequence (inclusive) with a default range and extra range.
 *
 * Only the range `defmin...defmax` is used when other defaulted parameters are set to extra values.
 * The range `min...max` is used only with core values from other defaulted parameters.
 * The two ranges are assumed not to overlap.
 */
template <typename S, typename T>
auto geometric(T min, T max, T step, T defmin, T defmax) {
    std::vector<T> v = {defmin};
    while (v.back() * step <= defmax) v.push_back(v.back() * step);
    size_t core_size = v.size();
    v.push_back(min);
    while (v.back() * step <= max) v.push_back(v.back() * step);
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = v[i];
        return true;
    }, core_size, v.size() - core_size);
}

/**
 * @brief Functor generating a recursively defined sequence.
 *
 * The recursive definition is given from three arguments:
 * - the list index `i` to be generated;
 * - the value previously generated `prev`;
 * - a \ref common::tagged_tuple "tagged_tuple" `tup` of parameters.
 * The recursive definition returns a `common::option<T>`, so that
 * `return {}` stops the recursion while `return v` provides a new item on the list.
 *
 * @param init A initialising value, to be fed to `f` for generating the first element.
 * @param f A function with signature `common::option<T>(size_t i, T prev, auto const& tup)`.
 */
template <typename S, typename T, typename F>
auto recursive(T init, F&& f) {
    std::vector<T> v;
    T prev = init;
    for (size_t i = 0; ; ++i) {
        common::option<T> r = f(i, prev);
        if (r.empty()) break;
        prev = r;
        v.push_back(r);
    }
    return details::make_generator<S, T>([=](auto& t, size_t i){
        common::get<S>(t) = v[i];
        return true;
    }, v.size(), 0);
}


//! @brief Functor generating a single tuple calculated from other values according to a given function.
template <typename S, typename T, typename F>
auto formula(F&& f) {
    return details::make_generator<S, T>([=](auto& t, size_t){
        common::get<S>(t) = f(t);
        return true;
    }, 1, 0);
}

//! @brief Functor generating a single tuple calculated as a representation of previously generated values.
template <typename S>
auto stringify(std::string prefix = "", std::string suffix = "") {
    return details::make_generator<S, std::string>([=](auto& t, size_t){
        using T = std::decay_t<decltype(t)>;
        constexpr size_t idx = T::tags::template find<S>;
        using R = common::tagged_tuple<typename T::tags::template slice<0, idx>, typename T::types::template slice<0, idx>>;
        common::get<S>(t) = prefix;
        R r = t;
        std::stringstream s;
        if (prefix != "")
            s << prefix << "_";
        r.print(s, common::underscore_tuple);
        if (suffix != "")
            s << "." << suffix;
        common::get<S>(t) = s.str();
        return true;
    }, 1, 0);
}

//! @brief Functor filtering out values from a sequence that match a given predicate.
template <typename F>
auto filter(F&& f) {
    return details::make_generator<>([=](auto& t, size_t){
        return not f(t);
    }, 1, 0);
}


/**
 * @brief Class generating a sequence of tagged tuples from generators.
 *
 * @tparam Gs The sequence of generator types for individual tags in the tuples, to be combined together.
 */
template <typename... Gs>
class tagged_tuple_sequence;

//! @brief Class generating a sequence of tagged tuples from generators (empty overload).
template <>
class tagged_tuple_sequence<> {
  public:
    //! @brief The tuple type that the class generates.
    using value_type = common::tagged_tuple_t<>;

    //! @brief Constructor setting up individual generators.
    tagged_tuple_sequence() {}

    //! @brief Returns the total size of the sequence generated (including filtered out values).
    inline size_t size() const {
        return 1;
    }

    //! @brief Function testing presence of an item of the sequence.
    inline bool count(size_t i) const {
        return i == 0;
    }

    //! @brief Function generating an item of the sequence.
    inline value_type operator[](size_t) const {
        return {};
    }

    /**
     * @brief Function generating and testing presence of an item of the sequence.
     *
     * @param[out] t The tuple in which to store the item generated.
     * @param[in]  i The index of the item to be generated.
     * @return     A boolean telling whether the given index has to be included (true) or skipped (false).
     */
    template <typename T>
    inline bool assign(T& t, size_t i) const {
        return true;
    }

  protected:
    //! @brief Returns the size of the core sequence that has to be expanded with every other value.
    inline size_t core_size() const {
        return 1;
    }

    //! @brief Returns the size of the extra sequence that should be expanded only with core values.
    inline size_t extra_size() const {
        return 0;
    }
};

//! @brief Class generating a sequence of tagged tuples from generators (recursive overload).
template <typename G, typename... Gs>
class tagged_tuple_sequence<G, Gs...> : public tagged_tuple_sequence<Gs...> {
  public:
    //! @brief The tuple type that the class generates.
    using value_type = common::tagged_tuple_cat<typename G::value_type, typename tagged_tuple_sequence<Gs...>::value_type>;

    //! @brief Constructor setting up the front generator.
    tagged_tuple_sequence(G const& g, tagged_tuple_sequence<Gs...> const& gs) :
        tagged_tuple_sequence<Gs...>(gs),
        m_core_extra_size(g.core_size() * tagged_tuple_sequence<Gs...>::extra_size()),
        m_extra_core_size(g.extra_size() * tagged_tuple_sequence<Gs...>::core_size()),
        m_core_size(g.core_size() * tagged_tuple_sequence<Gs...>::core_size()),
        m_extra_size(m_core_extra_size + m_extra_core_size),
        m_size(m_core_size + m_extra_size),
        m_generator(g) {}

    //! @brief Constructor setting up individual generators.
    tagged_tuple_sequence(G&& g, Gs&&... gs) :
        tagged_tuple_sequence<Gs...>(std::move(gs)...),
        m_core_extra_size(g.core_size() * tagged_tuple_sequence<Gs...>::extra_size()),
        m_extra_core_size(g.extra_size() * tagged_tuple_sequence<Gs...>::core_size()),
        m_core_size(g.core_size() * tagged_tuple_sequence<Gs...>::core_size()),
        m_extra_size(m_core_extra_size + m_extra_core_size),
        m_size(m_core_size + m_extra_size),
        m_generator(std::move(g)) {}

    //! @brief Returns the total size of the sequence generated (including filtered out values).
    inline size_t size() const {
        return m_size;
    }

    //! @brief Function testing presence of an item of the sequence.
    bool count(size_t i) const {
        if (i >= m_size) return false;
        value_type t;
        return assign(t, i);
    }

    //! @brief Function generating an item of the sequence.
    value_type operator[](size_t i) const {
        value_type t;
        assign(t, i);
        return t;
    }

    /**
     * @brief Function generating and testing presence of an item of the sequence.
     *
     * @param[out] t The tuple in which to store the item generated.
     * @param[in]  i The index of the item to be generated.
     * @return     A boolean telling whether the given index has to be included (true) or skipped (false).
     */
    template <typename T>
    bool assign(T& t, size_t i) const {
        if (i < m_core_size) {
            if (not m_generator(t, i / tagged_tuple_sequence<Gs...>::core_size())) return false;
            return tagged_tuple_sequence<Gs...>::assign(t, i % tagged_tuple_sequence<Gs...>::core_size());
        }
        i -= m_core_size;
        if (i < m_core_extra_size) {
            if (not m_generator(t, i / tagged_tuple_sequence<Gs...>::extra_size())) return false;
            return tagged_tuple_sequence<Gs...>::assign(t, i % tagged_tuple_sequence<Gs...>::extra_size() + tagged_tuple_sequence<Gs...>::core_size());
        }
        i -= m_core_extra_size;
        if (not m_generator(t, i / tagged_tuple_sequence<Gs...>::core_size() + m_generator.core_size())) return false;
        return tagged_tuple_sequence<Gs...>::assign(t, i % tagged_tuple_sequence<Gs...>::core_size());
    }

  protected:
    //! @brief Returns the size of the core sequence that has to be expanded with every other value.
    inline size_t core_size() const {
        return m_core_size;
    }

    //! @brief Returns the size of the extra sequence that should be expanded only with core values.
    inline size_t extra_size() const {
        return m_extra_size;
    }

  private:
    //! @brief The size of the sequence that is core for the first generator and extra for others.
    const size_t m_core_extra_size;
    //! @brief The size of the sequence that is extra for the first generator and core for others.
    const size_t m_extra_core_size;
    //! @brief The size of the core sequence that has to be expanded with every other value.
    const size_t m_core_size;
    //! @brief The size of the extra sequence that should be expanded only with core values.
    const size_t m_extra_size;
    //! @brief The total size of the sequence generated (including filtered out values).
    const size_t m_size;
    //! @brief The first generator.
    const G m_generator;
};

//! @brief Produces a generator of a sequence of tagged tuples, according to provided generators for individual tags.
template <typename... Gs>
inline auto make_tagged_tuple_sequence(Gs&&... gs) {
    return tagged_tuple_sequence<Gs...>(std::move(gs)...);
}

//! @brief Extends a generator of a sequence of tagged tuples, according to an additional provided generator for an individual tag.
template <typename G, typename... Gs>
inline auto extend_tagged_tuple_sequence(G const& g, tagged_tuple_sequence<Gs...> const& gs) {
    return tagged_tuple_sequence<G, Gs...>(g, gs);
}


/**
 * @brief Class concatenating multiple tagged tuple sequences.
 *
 * All sequences are assumed to have the same value type (modulo permutation).
 *
 * @tparam Ss The tagged tuple sequences.
 */
template <typename... Ss>
class tagged_tuple_sequences {
  public:
    //! @brief The tuple type that the class generates.
    using value_type = typename common::type_sequence<Ss...>::front::value_type;

    //! @brief Constructor setting up individual generators.
    tagged_tuple_sequences(Ss const&... ss) :
        m_shuffle(1),
        m_offset(0),
        m_stride(1),
        m_sequences(ss...) {
        std::array<size_t, sizeof...(Ss)> v = {ss.size()...};
        m_total_size = 0;
        for (size_t x : v) m_total_size += x;
        m_size = m_total_size;
    }

    //! @brief Returns the total size of the sequence generated (including filtered out values).
    inline size_t size() const {
        return m_size;
    }

    //! @brief Internally shuffles the sequence pseudo-randomly, in order to achieve better statistical balancing.
    void shuffle(uint_fast32_t seed = 0) {
        if (m_total_size < 3) return;
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dist(1, m_total_size-1);
        do m_shuffle = dist(gen);
        while (gcd(m_shuffle, m_total_size) > 1);
    }

    //! @brief Reduces the generator to a subsequence.
    void slice(size_t start, size_t end, size_t stride = 1) {
        m_offset = start;
        m_stride = stride;
        m_size = (std::min(end, m_total_size) - start + stride - 1) / stride;
    }

    //! @brief Returns an empty tuple of the type generated.
    inline value_type empty_tuple() const {
        return {};
    }

    //! @brief Function testing presence of an item of the sequence.
    inline bool count(size_t i) const {
        value_type t;
        return assign(t, i);
    }

    //! @brief Function generating an item of the sequence.
    inline value_type operator[](size_t i) const {
        value_type t;
        assign(t, i);
        return t;
    }

    /**
     * @brief Function generating and testing presence of an item of the sequence.
     *
     * @param[out] t The tuple in which to store the item generated.
     * @param[in]  i The index of the item to be generated.
     * @return     A boolean telling whether the given index has to be included (true) or skipped (false).
     */
    template <typename T>
    bool assign(T& t, size_t i) const {
        i = m_offset + m_stride * i;
        if (i >= m_total_size) return false;
        i = (i * m_shuffle) % m_total_size;
        return assign_impl(t, i, std::make_index_sequence<sizeof...(Ss)>{});
    }

  private:
    //! @brief Computes the gcd assuming that x > 0.
    size_t gcd(size_t x, size_t y) {
        do {
            y %= x;
            if (y == 0) return x;
            x %= y;
        } while (x > 0);
        return y;
    }

    //! @brief Accesses sequence elements iterating through the tuple of sequences (empty overload).
    template <typename T>
    inline bool assign_impl(T& t, size_t i, std::index_sequence<>) const {
        assert(false);
        return false;
    }

    //! @brief Accesses sequence elements iterating through the tuple of sequences (active overload).
    template <typename T, size_t j, size_t... js>
    inline bool assign_impl(T& t, size_t i, std::index_sequence<j, js...>) const {
        if (i < std::get<j>(m_sequences).size()) return std::get<j>(m_sequences).assign(t, i);
        return assign_impl(t, i - std::get<j>(m_sequences).size(), std::index_sequence<js...>{});
    }

    //! @brief The total size of the sequence generated (including filtered out values).
    size_t m_total_size;
    //! @brief The size of the reduced sequence after slicing.
    size_t m_size;
    //! @brief A factor shuffling the sequence order.
    size_t m_shuffle;
    //! @brief The offset to the first element of the sequence.
    size_t m_offset;
    //! @brief The step between consecutive elements of the sequence.
    size_t m_stride;
    //! @brief The first sequence.
    const std::tuple<Ss...> m_sequences;
};

//! @brief Produces a concatenation of multiple tagged tuple sequences.
template <typename... Ss>
inline auto make_tagged_tuple_sequences(Ss const&... ss) {
    return tagged_tuple_sequences<Ss...>(ss...);
}


//! @brief Tag identifying alternative template options for a network type (see \ref option_combine).
template <typename... Ts>
struct options {};

//! @cond INTERNAL
namespace details {
    //! @brief Enables if E is not a tagged tuple sequence or sequences.
    template <typename E, typename T = void>
    using ifn_sequence = common::ifn_class_template<tagged_tuple_sequence, E, common::ifn_class_template<tagged_tuple_sequences, E, T>>;

    //! @brief Converts a type into a type sequence.
    //! @{
    template <typename T>
    struct to_type_sequence {
        using type = common::type_sequence<T>;
    };
    template <typename... Ts>
    struct to_type_sequence<common::type_sequence<Ts...>> {
        using type = common::type_sequence<Ts...>;
    };
    template <typename T>
    using to_type_sequence_t = typename to_type_sequence<T>::type;
    //! @}

    //! @brief Manages options and non-options types.
    //! @{
    template <typename T>
    struct option_decay {
        using type = common::type_sequence<to_type_sequence_t<T>>;
    };
    template <typename... Ts>
    struct option_decay<options<Ts...>> {
        using type = common::type_sequence<to_type_sequence_t<Ts>...>;
    };
    template <typename T>
    using option_decay_t = typename option_decay<T>::type;
    //! @}

    //! @brief Maps a template to a sequence of options.
    //! @{
    template <template <class...> class C, typename T>
    struct map_template;
    template <template <class...> class C, typename... Ts>
    struct map_template<C, common::type_sequence<Ts...>> {
        using type = common::type_sequence<common::apply_templates<Ts, C>...>;
    };
    template <template <class...> class C, typename T>
    using map_template_t = typename map_template<C,T>::type;
    //! @}

    //! @brief Runs the \ref idx network type in the type sequence argument (empty overload).
    template <typename init_tuple>
    inline void network_run(common::type_sequence<>, size_t idx, init_tuple const& tup) {
        assert(false);
    }

    //! @brief Runs the \ref idx network type in the type sequence argument (active overload).
    template <typename T, typename... Ts, typename init_tuple>
    inline void network_run(common::type_sequence<T, Ts...>, size_t idx, init_tuple const& tup) {
        if (idx == 0) {
            typename T::net network{tup};
            network.run();
        } else network_run(common::type_sequence<Ts...>{}, idx-1, tup);
    }

    //! @brief Prints a sequence of network types (base overload).
    template <typename T>
    inline void print_types(common::type_sequence<T>) {
        std::cerr << common::type_name<T>();
    }

    //! @brief Prints a sequence of network types (recursive overload).
    template <typename T, typename S, typename... Ts>
    inline void print_types(common::type_sequence<T, S, Ts...>) {
        std::cerr << common::type_name<T>() << ", ";
        print_types(common::type_sequence<S, Ts...>{});
    }
}
//! @endcond

/**
 * @brief Instantiates a template for every possible combination from a given sequence of options.
 *
 * Fixed options can be given individually or grouped as type sequences.
 * Alternative options to be expanded in every possible combination have to be defined through the \ref options tag class.
 */
template <template <class...> class C, typename... Ts>
using option_combine = details::map_template_t<C, common::type_product<details::option_decay_t<Ts>...>>;


//! @brief Initialises MPI communication.
bool mpi_init(int& rank, int& n_procs);

//! @brief Forces MPI processes to wait for each other.
void mpi_barrier();

//! @brief Closes MPI communication.
void mpi_finalize();


/**
 *  @brief Runs a series of experiments with a non-distributed execution policy.
 *
 * @param x The network types to be run.
 * @param e An execution policy (see \ref common::tags::sequential_execution "sequential_execution", \ref common::tags::parallel_execution "parallel_execution", \ref common::tags::general_execution "general_execution", \ref common::tags::dynamic_execution "dynamic_execution").
 * @param vs Tagged tuple sequences used to initialise the various runs.
 */
template <typename... Ts, typename exec_t, typename... Ss>
common::ifn_among<exec_t, common::tags::distributed_execution, details::ifn_sequence<exec_t>>
run(common::type_sequence<Ts...> x, exec_t e, tagged_tuple_sequences<Ss...> vs) {
    details::print_types(x);
    std::cerr << ": running " << vs.size() << " simulations..." << std::flush;
    size_t p = 0;
    common::parallel_for(e, vs.size(), [&](size_t i, size_t t){
        if (t == 0 and i*100/vs.size() > p) {
            p = i*100/vs.size();
            std::cerr << p << "%..." << std::flush;
        }
        auto tup = vs.empty_tuple();
        if (vs.assign(tup, i)) details::network_run(x, common::get_or<tags::type_index>(tup, 0), tup);
    });
    std::cerr << "done." << std::endl;
}

#ifdef FCPP_MPI

//! @cond INTERNAL
namespace details {
    //! @brief Uses MPI to aggregate plots produced on different MPI processes.
    template <typename P>
    void aggregate_plots(P& p, int n_procs, int rank) {
        constexpr int rank_master = 0;
        if (rank == rank_master) {
            int size;
            int max_size = 128 * 1024 * 1024;
            char* buf = new char[max_size];
            MPI_Status status;
            for (int i = 1; i < n_procs; ++i) {
                P q;
                MPI_Recv(buf, max_size, MPI_CHAR, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_CHAR, &size);
                common::isstream is({buf, buf+size});
                is >> q;
                p += q;
            }
            delete [] buf;
        } else {
            common::osstream os;
            os << p;
            MPI_Send(os.data().data(), os.data().size(), MPI_CHAR, rank_master, 1, MPI_COMM_WORLD);
            p = P{};
        }
    }
} // details
//! @endcond

/**
 *  @brief Runs a series of experiments with a distributed execution policy through MPI.
 *
 * If FCPP_MPI is not defined, it raises an error with an assert.
 *
 * @param x The network types to be run.
 * @param e An execution policy (see \ref common::tags::distributed_execution "distributed_execution").
 * @param vs Tagged tuple sequences used to initialise the various runs.
 */
template <typename... Ts, typename... Ss>
void run(common::type_sequence<Ts...> x, common::tags::distributed_execution e, tagged_tuple_sequences<Ss...> vs) {
    // initialize mpi, generators and plotter address
    if (e.shuffle) vs.shuffle(42);
    auto plot = common::get_or<component::tags::plotter>(vs[0], nullptr);
    constexpr int rank_master = 0;
    int rank, n_procs;
    bool initialized = mpi_init(rank, n_procs);

    // setup initial chunks
    size_t initial_chunk = std::max(size_t((1 - e.dynamic) * vs.size()), std::min(vs.size(), e.num * n_procs));
    int pool_size = std::min(e.num, (initial_chunk + n_procs - 1) / n_procs);
    int istart = rank, i = istart, istep = n_procs;
    int rest = initial_chunk, iend = rest;
    int c = 0, p = 0, reqs = rest < vs.size() ? n_procs - 1  : 0;
    if (rank == rank_master) {
        details::print_types(x);
        std::cerr << ": running " << vs.size() << " simulations..." << std::flush;
    }

    // start working threads
    std::mutex m;
    std::vector<std::thread> pool;
    pool.reserve(pool_size);
    for (int t=0; t<pool_size; ++t)
        pool.emplace_back([&,t] () {
            size_t j;
            while (true) {
                m.lock();
                if (i < iend or i >= vs.size()) {   // there are things in local queue, grab one
                    j = i;
                    i = j + istep;
                } else if (rank == rank_master) {  // i am master, grab a whole chunk then one
                    istep = 1;
                    istart = rest + c * e.size;
                    iend = istart + e.size;
                    j = istart;
                    i = j + istep;
                    ++c;
                } else { // use MPI to ask for a chunk
                    if (rest + c * e.size < vs.size()) {
                        MPI_Send(&c, 0, MPI_INT, 0, 2, MPI_COMM_WORLD);
                        MPI_Recv(&c, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    istep = 1;
                    istart = rest + c * e.size;
                    iend = istart + e.size;
                    j = istart;
                    i = j + istep;
                }
                m.unlock();
                if (j >= vs.size()) break;
                int q = i < rest ? i : i * n_procs - (2 * istart + e.size) * (n_procs - 1) / 2;
                q = q * 100 / vs.size();
                if (rank == rank_master and t == 0 and q > p) {
                    p = q;
                    std::cerr << p << "%..." << std::flush;
                }
                auto tup = vs.empty_tuple();
                if (vs.assign(tup, j)) details::network_run(x, common::get_or<tags::type_index>(tup, 0), tup);
            }
            if (rank == rank_master and t == 0) std::cerr << "done." << std::endl;
        });

    // start MPI manager thread
    std::thread manager;
    if (rank == rank_master) manager = std::thread([&](){
        MPI_Status status;
        while (reqs > 0) {
            MPI_Recv(&c, 0, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
            int source = status.MPI_SOURCE;
            m.lock();
            MPI_Send(&c, 1, MPI_INT, source, 0, MPI_COMM_WORLD);
            ++c;
            m.unlock();
            if (rest + c * e.size >= vs.size()) --reqs;
        }
    });

    // wait threads to close and finalize
    for (std::thread& t : pool) t.join();
    if (rank == 0) manager.join();
    if (plot != nullptr)
        details::aggregate_plots(*plot, n_procs, rank);
    if (initialized) mpi_finalize();
}

#else

/**
 *  @brief Runs a series of experiments with a distributed execution policy through MPI.
 *
 * If FCPP_MPI is not defined, it raises an error with an assert.
 *
 * @param x The network types to be run.
 * @param e An execution policy (see \ref common::tags::distributed_execution "distributed_execution").
 * @param vs Tagged tuple sequences used to initialise the various runs.
 */
template <typename... Ts, typename... Ss>
inline void run(common::type_sequence<Ts...> x, common::tags::distributed_execution e, tagged_tuple_sequences<Ss...> vs) {
    assert(false);
}

#endif

//! @brief Runs a series of experiments (network types, explicit execution policy, sequence parameters).
template <typename... Ts, typename exec_t, typename... Gs, typename... Ss>
inline details::ifn_sequence<exec_t>
run(common::type_sequence<Ts...> x, exec_t e, tagged_tuple_sequence<Gs...> const& v, Ss const&... vs) {
    run(x, e, make_tagged_tuple_sequences(extend_tagged_tuple_sequence(arithmetic<tags::type_index>(0, int(sizeof...(Ts) - 1), 1), v),
                                          extend_tagged_tuple_sequence(arithmetic<tags::type_index>(0, int(sizeof...(Ts) - 1), 1), vs)...));
}

//!  @brief Runs a series of experiments (single network type, explicit execution policy)
template <typename T, typename exec_t, typename... Ss>
inline details::ifn_sequence<exec_t, common::ifn_class_template<common::type_sequence, T>>
run(T, exec_t e, Ss&&... vs) {
    run(common::type_sequence<T>{}, e, std::forward<Ss>(vs)...);
}

/**
 * @brief Runs a series of experiments (implicit execution policy, sequence parameters).
 *
 * If FCPP_MPI is defined, the execution policy is \ref common::tags::distributed_execution "distributed_execution", otherwise is \ref common::tags::dynamic_execution "dynamic_execution".
 */
template <typename T, typename... Gs, typename... Ss>
inline void run(T x, tagged_tuple_sequence<Gs...> const& v, Ss const&... vs) {
    run(x,
#ifdef FCPP_MPI
        common::tags::distributed_execution{},
#else
        common::tags::dynamic_execution{},
#endif
        v, vs...);
}

/**
 * @brief Runs a series of experiments (implicit execution policy, single sequences parameter).
 *
 * If FCPP_MPI is defined, the execution policy is \ref common::tags::distributed_execution "distributed_execution", otherwise is \ref common::tags::dynamic_execution "dynamic_execution".
 */
template <typename T, typename... Ss>
inline void run(T x, tagged_tuple_sequences<Ss...> vs) {
    run(x,
#ifdef FCPP_MPI
        common::tags::distributed_execution{},
#else
        common::tags::dynamic_execution{},
#endif
        vs);
}

//!  @brief Does not run a series of experiments (implicit execution policy, no parameters)
template <typename T>
inline void run(T) {}


} // batch


} // fcpp

#endif // FCPP_SIMULATION_BATCH_H_
