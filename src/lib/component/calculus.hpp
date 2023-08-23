// Copyright © 2023 Giorgio Audrito. All Rights Reserved.

/**
 * @file calculus.hpp
 * @brief Implementation of the `calculus` component providing the field calculus APIs.
 */

#ifndef FCPP_COMPONENT_CALCULUS_H_
#define FCPP_COMPONENT_CALCULUS_H_

#include <cassert>

#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "lib/common/serialize.hpp"
#include "lib/internal/context.hpp"
#include "lib/internal/trace.hpp"
#include "lib/internal/twin.hpp"
#include "lib/option/metric.hpp"
#include "lib/component/base.hpp"


/**
 * @brief Namespace containing all the objects in the FCPP library.
 */
namespace fcpp {


//! @cond INTERNAL
namespace details {
    //! @brief Accessess the context of a node.
    template <typename node_t>
    auto& get_context(node_t& node) {
        return node.m_context;
    }

    //! @brief Accessess the exports of a node.
    template <typename node_t>
    auto& get_export(node_t& node) {
        return node.m_export;
    }
}
//! @endcond


// Namespace for all FCPP components.
namespace component {


// Namespace of tags to be used for initialising components.
namespace tags {
    //! @brief Declaration tag associating to a sequence of types to be used in exports.
    template <typename... Ts>
    struct exports {};

    //! @brief Declaration tag associating to a callable class to be executed during rounds.
    template <typename T>
    struct program {};

    //! @brief Declaration tag associating to a metric class regulating the discard of exports.
    template <typename T>
    struct retain {};

    //! @brief Declaration flag associating to whether exports are wrapped in smart pointers.
    template <bool b>
    struct export_pointer {};

    //! @brief Declaration flag associating to whether exports for neighbours are split from those for self.
    template <bool b>
    struct export_split {};

    //! @brief Declaration flag associating to whether messages are dropped as they arrive (reduces memory footprint).
    template <bool b>
    struct online_drop {};

    //! @brief Node initialisation tag associating to the maximum size for a neighbourhood.
    struct hoodsize {};

    //! @brief Node initialisation tag associating to a threshold regulating discard of old messages.
    struct threshold {};
}


/**
 * @brief Component providing the field calculus APIs.
 *
 * <b>Declaration tags:</b>
 * - \ref tags::exports defines a sequence of types to be used in exports (defaults to the empty sequence).
 * - \ref tags::program defines a callable class to be executed during rounds (defaults to \ref calculus::null_program).
 * - \ref tags::retain defines a metric class regulating the discard of exports (defaults to \ref metric::once).
 *
 * <b>Declaration flags:</b>
 * - \ref tags::export_pointer defines whether exports are wrapped in smart pointers (defaults to \ref FCPP_EXPORT_PTR).
 * - \ref tags::export_split defines whether exports for neighbours are split from those for self (defaults to \ref FCPP_EXPORT_NUM `== 2`).
 * - \ref tags::online_drop defines whether messages are dropped as they arrive (defaults to \ref FCPP_ONLINE_DROP).
 *
 * <b>Node initialisation tags:</b>
 * - \ref tags::hoodsize associates to the maximum number of neighbours allowed (defaults to `std::numeric_limits<device_t>::%max()`).
 * - \ref tags::threshold associates to a `T::result_type` threshold (where `T` is the class specified with \ref tags::retain) regulating discarding of old messages (defaults to the result of `T::build()`).
 *
 * Retain classes should (see \ref metric for a list of available ones):
 * - provide a `result_type` type member which has to be totally ordered, for example:
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 *   using result_type = real_t;
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~
 * - be able to build a default `result_type` to be used as threshold:
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 *   result_type build();
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~
 * - be able to build a `result_type` from a `tagged_tuple` message possibly using node data:
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 *   template <typename N, typename S, typename T>
 *   result_type build(N const& node, times_t t, device_t d, common::tagged_tuple<S,T> const& m);
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~
 * - be able to update by comparing a `result_type` with node data:
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 *   template <typename N>
 *   result_type update(result_type const&, N const& node);
 *   ~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Round classes should be default-constructible and be callable with the following signature:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 * template <typename node_t>
 * void operator()(node_t& node, times_t t);
 * ~~~~~~~~~~~~~~~~~~~~~~~~~
 */
template <class... Ts>
struct calculus {
    //! @brief Callable class performing no operation.
    struct null_program {
        //! @brief Call operator doing nothing.
        template <typename node_t>
        void operator()(node_t&, times_t) {}
    };

    //! @brief Callable class to be executed during rounds.
    using program_type = common::option_type<tags::program, null_program, Ts...>;

    //! @brief Metric class regulating the discard of exports.
    using retain_type = common::option_type<tags::retain, metric::once, Ts...>;

    //! @brief Sequence of types to be used in exports.
    using exports_type = common::export_list<common::option_types<tags::exports, Ts...>>;

    //! @brief Whether exports are wrapped in smart pointers.
    constexpr static bool export_pointer = common::option_flag<tags::export_pointer, FCPP_EXPORT_PTR, Ts...>;

    //! @brief Whether exports for neighbours are split from those for self.
    constexpr static bool export_split = common::option_flag<tags::export_split, FCPP_EXPORT_NUM == 2, Ts...>;

    //! @brief Whether messages are dropped as they arrive.
    constexpr static bool online_drop = common::option_flag<tags::online_drop, FCPP_ONLINE_DROP, Ts...>;

    /**
     * @brief The actual component.
     *
     * Component functionalities are added to those of the parent by inheritance at multiple levels: the whole component class inherits tag for static checks of correct composition, while `node` and `net` sub-classes inherit actual behaviour.
     * Further parametrisation with F enables <a href="https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern">CRTP</a> for static emulation of virtual calls.
     *
     * @param F The final composition of all components.
     * @param P The parent component to inherit from.
     */
    template <typename F, typename P>
    struct component : public P {
        //! @cond INTERNAL
        DECLARE_COMPONENT(calculus);
        //! @endcond

        //! @brief The local part of the component.
        class node : public P::node {
            //! @cond INTERNAL
            //! @brief Friendship declarations.
            template <typename node_t>
            friend auto& fcpp::details::get_context(node_t& node);
            template <typename node_t>
            friend auto& fcpp::details::get_export(node_t& node);
            //! @endcond

          public: // visible by net objects and the main program
            //! @brief The type of the metric on exports.
            using metric_type = typename retain_type::result_type;

            //! @brief The type of the context of exports from other devices.
            using context_type = internal::context_t<online_drop, export_pointer, metric_type, exports_type>;

            //! @brief The type of the exports of the current device.
            using export_type = typename context_type::export_type;

            //! @brief Helper type providing access to the context for self-messages.
            template <typename A>
            struct self_context_type {
                //! @brief Inserts a value into the exports.
                inline void insert(A x) {
                    assert(n.m_export.first()->template count<A>(t) == 0);
                    n.m_export.first()->template insert<A>(t, std::move(x));
                }

                //! @brief Accesses old stored values given a default.
                inline A const& old(A const& def) {
                    return n.m_context.first().template old<A>(t, def, n.uid);
                }

              private:
                //! @brief Private constructor.
                self_context_type(node& n, trace_t t) : n(n), t(t) {}
                //! @brief Friendship declaration to allow construction from nodes.
                friend class node;
                //! @brief A reference to the node object.
                node& n;
                //! @brief The current stack trace hash.
                trace_t t;
            };

            //! @brief Helper type providing access to the context for neighbour messages.
            template <typename A>
            struct nbr_context_type {
                //! @brief Inserts a value into the exports.
                inline void insert(A x) {
                    assert(n.m_export.second()->template count<A>(t) == 0);
                    n.m_export.second()->template insert<A>(t, std::move(x));
                }

                //! @brief Accesses old stored values given a default.
                inline A const& old(A const& def) {
                    return n.m_context.second().template old<A>(t, def, n.uid);
                }

                //! @brief Accesses old stored values given a default.
                inline to_field<A> nbr(A const& def) {
                    return n.m_context.second().template nbr<A>(t, def, n.uid);
                }

              private:
                //! @brief Private constructor.
                nbr_context_type(node& n, trace_t t) : n(n), t(t) {}
                //! @brief Friendship declaration to allow construction from nodes.
                friend class node;
                //! @brief A reference to the node object.
                node& n;
                //! @brief The current stack trace hash.
                trace_t t;
            };

            //! @brief Helper type providing access to the context for neighbour call points.
            struct void_context_type {
                //! @brief Accesses the list of devices aligned with the call point.
                inline std::vector<device_t> align() {
                    n.m_export.second()->insert(t);
                    return n.m_context.second().align(t, n.uid);
                }

              private:
                //! @brief Private constructor.
                void_context_type(node& n, trace_t t) : n(n), t(t) {}
                //! @brief Friendship declaration to allow construction from nodes.
                friend class node;
                //! @brief A reference to the node object.
                node& n;
                //! @brief The current stack trace hash.
                trace_t t;
            };

            //! @brief A `tagged_tuple` type used for messages to be exchanged with neighbours.
            using message_t = typename P::node::message_t::template push_back<calculus_tag, export_type>;

            /**
             * @brief Main constructor.
             *
             * @param n The corresponding net object.
             * @param t A `tagged_tuple` gathering initialisation values.
             */
            template <typename S, typename T>
            node(typename F::net& n, common::tagged_tuple<S,T> const& t) : P::node(n,t), m_context{}, m_metric{t}, m_hoodsize{common::get_or<tags::hoodsize>(t, std::numeric_limits<device_t>::max())}, m_threshold{common::get_or<tags::threshold>(t, m_metric.build())} {}

            //! @brief Performs computations at round start with current time `t`.
            void round_start(times_t t) {
                P::node::round_start(t);
                assert(stack_trace.empty());
                m_context.second().freeze(m_hoodsize, P::node::uid);
                m_export = {};
                std::vector<device_t> nbr_ids = m_context.second().align(P::node::uid);
                std::vector<device_t> nbr_vals;
                nbr_vals.emplace_back();
                nbr_vals.insert(nbr_vals.end(), nbr_ids.begin(), nbr_ids.end());
                m_nbr_uid = fcpp::details::make_field(std::move(nbr_ids), std::move(nbr_vals));
            }

            //! @brief Performs computations at round middle with current time `t`.
            void round_main(times_t t) {
                P::node::round_main(t);
                m_callback(P::node::as_final(), t);
            }

            //! @brief Performs computations at round end with current time `t`.
            void round_end(times_t t) {
                assert(stack_trace.empty());
                P::node::round_end(t);
                m_context.second().unfreeze(P::node::as_final(), m_metric, m_threshold);
            }

            //! @brief Receives an incoming message (possibly reading values from sensors).
            template <typename S, typename T>
            void receive(times_t t, device_t d, common::tagged_tuple<S,T> const& m) {
                P::node::receive(t, d, m);
                m_context.second().insert(d, common::get<calculus_tag>(m), m_metric.build(P::node::as_final(), t, d, m), m_threshold, m_hoodsize);
                if (export_split and d == P::node::uid)
                    m_context.first().insert(d, m_export.first(), m_metric.build(P::node::as_final(), t, d, m), m_threshold, m_hoodsize);
            }

            //! @brief Produces the message to send, both storing it in its argument and returning it.
            template <typename S, typename T>
            common::tagged_tuple<S,T>& send(times_t t, common::tagged_tuple<S,T>& m) const {
                P::node::send(t, m);
                common::get<calculus_tag>(m) = m_export.second();
                return m;
            }

            //! @brief Total number of neighbours (including self and those not aligned).
            size_t size() const {
                return m_context.second().size(P::node::uid);
            }

            //! @brief Identifiers of the neighbours.
            field<device_t> const& nbr_uid() const {
                return m_nbr_uid;
            }

            //! @brief Accesses the threshold for message retain.
            metric_type message_threshold() const {
                return m_threshold;
            }

            //! @brief Modifies the threshold for message retain.
            void message_threshold(metric_type t) {
                m_threshold = t;
            }

            //! @brief Accesses the context for self-messages.
            template <typename A>
            self_context_type<A> self_context(trace_t call_point) {
                return {*this, stack_trace.hash(call_point)};
            }

            //! @brief Accesses the context for neighbour messages.
            template <typename A>
            nbr_context_type<A> nbr_context(trace_t call_point) {
                return {*this, stack_trace.hash(call_point)};
            }

            //! @brief Accesses the context for neighbour call points.
            void_context_type void_context(trace_t call_point) {
                return {*this, stack_trace.hash(call_point)};
            }

            //! @brief Stack trace maintained during aggregate function execution.
            internal::trace stack_trace;

          private: // implementation details
            //! @brief Map associating devices to their exports (`first` for local device, `second` for others).
            internal::twin<context_type, not export_split> m_context;

            //! @brief Exports of the current device (`first` for local device, `second` for others).
            internal::twin<export_type, not export_split> m_export;

            //! @brief The callable class representing the main round.
            program_type m_callback;

            //! @brief The metric class.
            retain_type m_metric;

            //! @brief Maximum amount of neighbours allowed.
            device_t m_hoodsize;

            //! @brief Maximum export metric value allowed.
            metric_type m_threshold;

            //! @brief Identifiers of the neighbours.
            field<device_t> m_nbr_uid;
        };

        //! @brief The global part of the component.
        using net = typename P::net;
    };
};


}


//! @name field operators
//! @{

//! @brief Computes the restriction of a local to the current domain.
template <typename node_t, typename A, typename = if_local<A>>
inline A align(node_t const&, trace_t, A&& x) {
    return x;
}

//! @brief Computes the restriction of a field to the current domain.
template <typename node_t, typename A, typename = if_field<A>>
A align(node_t& node, trace_t call_point, A const& x) {
    auto ctx = node.void_context(call_point);
    return details::align(x, ctx.align());
}

//! @brief Computes the restriction of a field to the current domain.
template <typename node_t, typename A, typename = if_field<A>, typename = std::enable_if_t<not std::is_reference<A>::value>>
A align(node_t& node, trace_t call_point, A&& x) {
    auto ctx = node.void_context(call_point);
    return details::align(std::move(x), ctx.align());
}

//! @brief Computes in-place the restriction of a field to the current domain.
template <typename node_t, typename A, typename = if_local<A>>
inline void align_inplace(node_t const&, trace_t, A&) {}

//! @brief Computes in-place the restriction of a field to the current domain.
template <typename node_t, typename A, typename = if_field<A>>
void align_inplace(node_t& node, trace_t call_point, A& x) {
    auto ctx = node.void_context(call_point);
    details::align(x, ctx.align());
}

//! @brief Accesses the local value of a field.
template <typename node_t, typename A>
to_local<A const&> self(node_t const& node, trace_t, A const& x) {
    return details::self(x, node.uid);
}

//! @brief Accesses the local value of a field (moving).
template <typename node_t, typename A, typename = std::enable_if_t<not std::is_reference<A>::value>>
to_local<A&&> self(node_t const& node, trace_t, A&& x) {
    return details::self(std::move(x), node.uid);
}

//! @brief Accesses a given value of a field.
template <typename node_t, typename A>
to_local<A const&> self(node_t const&, trace_t, A const& x, device_t uid) {
    return details::self(x, uid);
}

//! @brief Accesses a given value of a field (moving).
template <typename node_t, typename A, typename = std::enable_if_t<not std::is_reference<A>::value>>
to_local<A&&> self(node_t const&, trace_t, A&& x, device_t uid) {
    return details::self(std::move(x), uid);
}

//! @brief Returns the local value of a field (modifiable).
template <typename node_t, typename A>
to_local<A&> mod_self(node_t const& node, trace_t, A& x) {
    return details::self(x, node.uid);
}

//! @brief Modifies the local value of a field.
template <typename node_t, typename A, typename B>
to_field<std::decay_t<A>> mod_self(node_t const& node, trace_t, A&& x, B&& y) {
    return details::mod_self(std::forward<A>(x), std::forward<B>(y), node.uid);
}

//! @brief Accesses the default value of a field.
template <typename node_t, typename A>
to_local<A const&> other(node_t const&, trace_t, A const& x) {
    return details::other(x);
}

//! @brief Accesses the default value of a field (moving).
template <typename node_t, typename A, typename = std::enable_if_t<not std::is_reference<A>::value>>
to_local<A&&> other(node_t const&, trace_t, A&& x) {
    return details::other(std::move(x));
}

//! @brief Returns the default value of a field (modifiable, ensuring alignment).
template <typename node_t, typename A, typename = if_field<A>>
to_local<A&> mod_other(node_t& node, trace_t call_point, A& x) {
    auto ctx = node.void_context(call_point);
    return details::other(details::align_inplace(x, ctx.align()));
}

//! @brief Modifies the local value of a field (ensuring alignment).
template <typename node_t, typename A, typename B>
to_field<std::decay_t<A>> mod_other(node_t& node, trace_t call_point, A const& x, B const& y) {
    auto ctx = node.void_context(call_point);
    return details::mod_other(x, y, ctx.align());
}

//! @brief Reduces a field to a single value by a binary operation.
template <typename node_t, typename O, typename A>
auto fold_hood(node_t& node, trace_t call_point, O&& op, A const& a) {
    auto ctx = node.void_context(call_point);
    return details::fold_hood(op, a, ctx.align());
}

//! @brief Reduces a field to a single value by a binary operation with a given value for self.
template <typename node_t, typename O, typename A, typename B>
auto fold_hood(node_t& node, trace_t call_point, O&& op, A const& a, B const& b) {
    auto ctx = node.void_context(call_point);
    return details::fold_hood(op, a, b, ctx.align(), node.uid);
}

//! @brief Computes the number of neighbours aligned to the current call point.
template <typename node_t>
size_t count_hood(node_t& node, trace_t call_point) {
    auto ctx = node.void_context(call_point);
    return ctx.align().size();
}

//! @brief Computes the identifiers of neighbours aligned to the current call point.
template <typename node_t>
field<device_t> nbr_uid(node_t& node, trace_t call_point) {
    auto ctx = node.void_context(call_point);
    std::vector<device_t> ids = ctx.align();
    std::vector<device_t> vals;
    vals.emplace_back();
    vals.insert(vals.end(), ids.begin(), ids.end());
    return details::make_field(std::move(ids), std::move(vals));
}

//! @}


//! @cond INTERNAL
namespace details {
    template <typename D, typename T>
    struct result_unpack {
        using type = std::enable_if_t<std::is_convertible<D, T>::value, common::type_sequence<T, T>>;
    };
    template <typename D, typename R, typename A>
    struct result_unpack<D, tuple<R, A>> {
        using type = std::conditional_t<
            std::is_convertible<D, tuple<R, A>>::value,
            common::type_sequence<tuple<R, A>, tuple<R, A>>,
            std::enable_if_t<std::is_convertible<D, tuple<R, A>>::value or std::is_convertible<D, A>::value, common::type_sequence<R, A>>
        >;
    };

    template <typename D, typename R, typename A, typename = std::enable_if_t<std::is_convertible<D, A>::value>>
    inline R&& maybe_first(common::type_sequence<D>, tuple<R,A>& t) {
        return std::move(get<0>(t));
    }
    template <typename D, typename R, typename A, typename = std::enable_if_t<std::is_convertible<D, A>::value>>
    inline A&& maybe_second(common::type_sequence<D>, tuple<R,A>& t) {
        return std::move(get<1>(t));
    }
    template <typename D, typename T, typename = std::enable_if_t<std::is_convertible<D, T>::value>>
    inline T&& maybe_first(common::type_sequence<D>, T& x) {
        return std::move(x);
    }
    template <typename D, typename T, typename = std::enable_if_t<std::is_convertible<D, T>::value>>
    inline T& maybe_second(common::type_sequence<D>, T& x) {
        return x;
    }
}
//! @endcond

//! @brief The data type returned by an update function call T given default of type D.
template <typename D, typename T>
using return_result_type = typename details::result_unpack<std::decay_t<D>, std::decay_t<std::result_of_t<T>>>::type::front;

//! @brief The export type written by an update function call T given default of type D.
template <typename D, typename T>
using export_result_type = typename details::result_unpack<std::decay_t<D>, std::decay_t<std::result_of_t<T>>>::type::back;


//! @name old-based coordination operators
//! @{
/**
 * @brief The previous-round value (defaults to first argument), modified through the second argument.
 *
 * Corresponds to the `rep` construct of the field calculus.
 * The \p op argument may return a `A` or a `tuple<R,A>`, where `A` is
 * compatible with the default type `D`. In the latter case,
 * the first element of the returned pair is returned by the function, while
 * the second element of the returned pair is written in the exports.
 */
template <typename node_t, typename D, typename G>
return_result_type<D, G(D)> old(node_t& node, trace_t call_point, D const& f0, G&& op) {
    using A = export_result_type<D, G(D)>;
    auto ctx = node.template self_context<A>(call_point);
    auto f = op(align(node, call_point, ctx.old(f0)));
    ctx.insert(details::maybe_second(common::type_sequence<D>{}, f));
    return details::maybe_first(common::type_sequence<D>{}, f);
}
/**
 * @brief The previous-round value of the second argument, defaulting to the first argument if no previous value.
 *
 * Equivalent to:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 * old(f0, [](A fo){
 *     return make_tuple(fo, f);
 * })
 * ~~~~~~~~~~~~~~~~~~~~~~~~~
 */
template <typename node_t, typename D, typename A, typename = std::enable_if_t<std::is_convertible<D, A>::value>>
A old(node_t& node, trace_t call_point, D const& f0, A const& f) {
    auto ctx = node.template self_context<A>(call_point);
    ctx.insert(f);
    return align(node, call_point, ctx.old(f0));
}
/**
 * @brief The previous-round value of the argument.
 *
 * Equivalent to `old(f, f)`.
 */
template <typename node_t, typename A>
inline A old(node_t& node, trace_t call_point, A const& f) {
    return old(node, call_point, f, f);
}

//! @brief The exports type used by the old construct with message type `T`.
template <typename T>
using old_t = common::export_list<T>;
//! @}


//! @name nbr-based coordination operators
//! @{
/**
 * @brief The neighbours' value of the result (defaults to first argument), modified through the second argument.
 *
 * Corresponds to the `share` construct of the field calculus.
 * The \p op argument may return a `A` or a `tuple<R,A>`, where `A` is
 * compatible with the default type `D`. In the latter case,
 * the first element of the returned pair is returned by the function, while
 * the second element of the returned pair is written in the exports.
 */
template <typename node_t, typename D, typename G>
return_result_type<D, G(to_field<D>)> nbr(node_t& node, trace_t call_point, D const& f0, G&& op) {
    using A = export_result_type<D, G(to_field<D>)>;
    auto ctx = node.template nbr_context<A>(call_point);
    auto f = op(ctx.nbr(f0));
    ctx.insert(details::maybe_second(common::type_sequence<D>{}, f));
    return details::maybe_first(common::type_sequence<D>{}, f);
}
/**
 * @brief The neighbours' value of the second argument, defaulting to the first argument.
 *
 * Equivalent to:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 * nbr(f0, [](to_field<A> fn){
 *     return std::make_pair(fn, f);
 * })
 * ~~~~~~~~~~~~~~~~~~~~~~~~~
 */
template <typename node_t, typename D, typename A, typename = std::enable_if_t<std::is_convertible<D, A>::value>>
to_field<A> nbr(node_t& node, trace_t call_point, D const& f0, A const& f) {
    auto ctx = node.template nbr_context<A>(call_point);
    ctx.insert(f);
    return ctx.nbr(f0);
}
/**
 * @brief The neighbours' value of the argument.
 *
 * Equivalent to `nbr(f, f)`.
 */
template <typename node_t, typename A>
inline to_field<A> nbr(node_t& node, trace_t call_point, A const& f) {
    return nbr(node, call_point, f, f);
}

//! @brief The exports type used by the nbr construct with message type `T`.
template <typename T>
using nbr_t = common::export_list<T>;
//! @}


//! @name mixed coordination operators
//! @{
/**
 * @brief The result of the second argument given info from neighbours' and self.
 *
 * The \p op argument may return a `A` or a `tuple<B,A>`. In the latter case,
 * the first element of the returned pair is returned by the function, while
 * the second element of the returned pair is written in the exports.
 */
template <typename node_t, typename D, typename G>
return_result_type<D, G(D, to_field<D>)> oldnbr(node_t& node, trace_t call_point, D const& f0, G&& op) {
    using A = export_result_type<D, G(D, to_field<D>)>;
    auto ctx = node.template nbr_context<A>(call_point);
    auto f = op(align(node, call_point, ctx.old(f0)), ctx.nbr(f0));
    ctx.insert(details::maybe_second(common::type_sequence<D>{}, f));
    return details::maybe_first(common::type_sequence<D>{}, f);
}

//! @brief The exports type used by the oldnbr construct with message type `T`.
template <typename T>
using oldnbr_t = common::export_list<T>;
//! @}


//! @brief Executes code independently in a partition of the network based on the value of a given key.
template <typename node_t, typename T, typename G>
auto split(node_t& node, trace_t call_point, T&& key, G&& f) {
    internal::trace_call trace_caller(node.stack_trace, call_point);
    internal::trace_key trace_process(node.stack_trace, common::hash_to<trace_t>(std::forward<T>(key)));
    return f();
}
//! @brief The exports type used by the split construct.
using split_t = common::export_list<>;


//! @name aggregate processes operators
//! @{
/**
 * @brief The status of an aggregate process in a node.
 *
 * The values mean:
 * - Termination is propagated to neighbour nodes in order to ensure the process ends.
 * - An external node is not part of the aggregate process, and its exports cannot be seen by neighbours (deprecated).
 * - A border node is part of the process, but does not cause the process to expand to neighbours.
 * - An internal node is part of the process and propagates it to neighbours.
 * - Every status may request to return the output or not to the `spawn` caller.
 *
 * Note that `status::output` is provided as a synonym of `status::internal_output`, and
 * `status::x and status::output` equals `status::x_output`.
 */
enum class status : char { terminated, external_deprecated, border, internal, terminated_output, external_output_deprecated, border_output, internal_output, output };

//! @brief String representation of a status.
std::string to_string(status);

//! @brief Printing status.
template <typename O>
O& operator<<(O& o, status s) {
    o << to_string(s);
    return o;
}

//! @brief Merges the output status with another status (undefined for other combinations of statuses).
inline constexpr status operator&&(status x, status y) {
    if (y == status::output) {
        assert(x != status::output);
        return static_cast<status>(static_cast<char>(x) | char(4));
    }
    if (x == status::output) {
        assert(y != status::output);
        return static_cast<status>(static_cast<char>(y) | char(4));
    }
    assert(false);
    return status::output;
}

//! @brief Removes the output status from another status (undefined for other combinations of statuses).
inline constexpr status operator^(status x, status y) {
    if (y == status::output) {
        assert(x != status::output);
        return static_cast<status>(static_cast<char>(x) & char(3));
    }
    if (x == status::output) {
        assert(y != status::output);
        return static_cast<status>(static_cast<char>(y) & char(3));
    }
    assert(false);
    return status::output;
}

//! @brief Handles a process, spawning instances of it for every key in the `key_set` and passing general arguments `xs` (overload with boolean status corresponding to `status::internal_output` and `status::border_output`).
template <typename node_t, typename G, typename S, typename... Ts, typename K = typename std::decay_t<S>::value_type, typename T = std::decay_t<std::result_of_t<G(K const&, Ts const&...)>>, typename R = std::decay_t<tuple_element_t<0,T>>, typename B = std::decay_t<tuple_element_t<1,T>>>
std::enable_if_t<std::is_same<B,bool>::value, std::unordered_map<K, R, common::hash<K>>>
spawn(node_t& node, trace_t call_point, G&& process, S&& key_set, Ts const&... xs) {
    using keyset_t = std::unordered_set<K, common::hash<K>>;
    using resmap_t = std::unordered_map<K, R, common::hash<K>>;
    auto ctx = node.template nbr_context<keyset_t>(call_point);
    field<keyset_t> fk = ctx.nbr({});
    // keys to be propagated and terminated
    keyset_t ky(key_set.begin(), key_set.end()), km;
    for (size_t i = 1; i < details::get_vals(fk).size(); ++i)
        ky.insert(details::get_vals(fk)[i].begin(), details::get_vals(fk)[i].end());
    internal::trace_call trace_caller(node.stack_trace, call_point);
    resmap_t rm;
    // run process for every gathered key
    for (K const& k : ky) {
        internal::trace_key trace_process(node.stack_trace, common::hash_to<trace_t>(k));
        bool b;
        tie(rm[k], b) = process(k, xs...);
        // if true status, propagate key to neighbours
        if (b) km.insert(k);
    }
    ctx.insert(km);
    return rm;
}

/**
 * @brief Handles a process, spawning instances of it for every key in the `key_set` and passing general arguments `xs` (overload with general status).
 *
 * Does not support the "external" status, which is treated equally as "border".
 * Termination propagates causing devices to get into "border" status.
 */
template <typename node_t, typename G, typename S, typename... Ts, typename K = typename std::decay_t<S>::value_type, typename T = std::decay_t<std::result_of_t<G(K const&, Ts const&...)>>, typename R = std::decay_t<tuple_element_t<0,T>>, typename B = std::decay_t<tuple_element_t<1,T>>>
std::enable_if_t<std::is_same<B,status>::value, std::unordered_map<K, R, common::hash<K>>>
spawn(node_t& node, trace_t call_point, G&& process, S&& key_set, Ts const&... xs) {
    using keymap_t = std::unordered_map<K, B, common::hash<K>>;
    using resmap_t = std::unordered_map<K, R, common::hash<K>>;
    auto ctx = node.template nbr_context<keymap_t>(call_point);
    // keys to be propagated and terminated
    std::unordered_set<K, common::hash<K>> ky(key_set.begin(), key_set.end()), kn;
    for (auto const& m : details::get_vals(ctx.nbr({})))
        for (auto const& k : m) {
            if (k.second == status::terminated)
                kn.insert(k.first);
            else
                ky.insert(k.first);
        }
    internal::trace_call trace_caller(node.stack_trace, call_point);
    keymap_t km;
    resmap_t rm;
    // run process for every gathered key
    for (K const& k : ky)
        if (kn.count(k) == 0) {
            internal::trace_key trace_process(node.stack_trace, common::hash_to<trace_t>(k));
            R r;
            status s;
            tie(r, s) = process(k, xs...);
            // if output status, add result to returned map
            if ((char)s >= 4) {
                rm.emplace(k, std::move(r));
                s = s == status::output ? status::internal : static_cast<status>((char)s & char(3));
            }
            // if internal or terminated, propagate key status to neighbours
            if (s == status::terminated or s == status::internal)
                km.emplace(k, s);
        } else km.emplace(k, status::terminated);
    ctx.insert(km);
    return rm;
}

//! @brief The exports type used by the spawn construct with key type `K` and status type `B`.
template <typename K, typename B>
using spawn_t = common::export_list<std::conditional_t<std::is_same<B, bool>::value, std::unordered_set<K, common::hash<K>>, std::unordered_map<K, B, common::hash<K>>>>;
//! @}


}

#endif // FCPP_COMPONENT_CALCULUS_H_
