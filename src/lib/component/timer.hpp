// Copyright © 2023 Giorgio Audrito. All Rights Reserved.

/**
 * @file timer.hpp
 * @brief Implementation of the `timer` component managing round executions.
 */

#ifndef FCPP_COMPONENT_TIMER_H_
#define FCPP_COMPONENT_TIMER_H_

#include <cassert>
#include <type_traits>

#include "lib/component/base.hpp"
#include "lib/data/field.hpp"
#include "lib/option/sequence.hpp"


/**
 * @brief Namespace containing all the objects in the FCPP library.
 */
namespace fcpp {


// Namespace for all FCPP components.
namespace component {


// Namespace of tags to be used for initialising components.
namespace tags {
    //! @brief Declaration tag associating to a generator of delays for reactive rounds after a message is received (defaults to \ref sequence::never).
    template <typename T>
    struct reactive_delay {};

    //! @brief Declaration tag associating to a list of sequence generator type scheduling rounds (defaults to \ref sequence::never).
    template <typename... Ts>
    struct round_schedule {};

    //! @brief Node initialisation tag associating to a starting time of execution (defaults to \ref TIME_MAX).
    struct start {};
}


/**
 * @brief Component managing and scheduling round executions.
 *
 * If a \ref randomizer parent component is not found, \ref crand is used as random generator.
 *
 * <b>Declaration tags:</b>
 * - \ref tags::reactive_delay defines a generator of delays for reactive rounds after a message is received (defaults to \ref sequence::never).
 * - \ref tags::round_schedule defines a list of sequence generator types scheduling rounds (defaults to \ref sequence::never).
 *
 * <b>Node initialisation tags:</b>
 * - \ref tags::start associates to a starting time of execution (defaults to \ref TIME_MAX).
 */
template <class... Ts>
struct timer {
    //! @brief Generator type for the delay of reactive rounds after messages.
    using reactive_delay_type = common::option_type<tags::reactive_delay, sequence::never, Ts...>;

    //! @brief Sequence generator type scheduling rounds.
    using schedule_type = sequence::merge_t<common::option_types<tags::round_schedule, Ts...>>;

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
        DECLARE_COMPONENT(timer);
        CHECK_COMPONENT(calculus);
        CHECK_COMPONENT(identifier);
        CHECK_COMPONENT(randomizer);
        //! @endcond

        //! @brief The local part of the component.
        class node : public P::node {
          public: // visible by net objects and the main program
            /**
             * @brief Main constructor.
             *
             * @param n The corresponding net object.
             * @param t A `tagged_tuple` gathering initialisation values.
             */
            template <typename S, typename T>
            node(typename F::net& n, common::tagged_tuple<S,T> const& t) : P::node(n,t), m_neigh(TIME_MIN), m_react_gen(get_generator(has_randomizer<P>{}, *this),t), m_schedule(get_generator(has_randomizer<P>{}, *this),t) {
                m_prev = m_cur = TIME_MIN;
                m_next = common::get_or<tags::start>(t, TIME_MAX);
                m_offs = (m_next == TIME_MAX ? 0 : m_next);
                if (m_schedule.next() < TIME_MAX)
                    m_next = m_offs + m_schedule(get_generator(has_randomizer<P>{}, *this), common::tagged_tuple_t<>{});
                m_mod_next = std::min(m_next, TIME_FAR);
                m_react = m_react_gen(get_generator(has_randomizer<P>{}, *this), common::tagged_tuple_t<>{});
                m_fact = 1;
            }

            /**
             * @brief Returns next event to schedule for the node component.
             *
             * Should correspond to the next time also during updates.
             */
            times_t next() const {
                return std::min(m_mod_next, P::node::next());
            }

            //! @brief Updates the internal status of node component.
            void update() {
                if (m_next < P::node::next()) {
                    fcpp::details::self(m_neigh, P::node::uid) = m_cur;
                    m_prev = m_cur;
                    m_cur = m_next;
                    m_next = m_schedule(get_generator(has_randomizer<P>{}, *this), common::tagged_tuple_t<>{});
                    m_next = m_offs < TIME_MAX and m_next < TIME_MAX ? m_next/m_fact + m_offs : TIME_MAX;
                    m_mod_next = std::min(m_next, std::max(TIME_FAR, m_offs));
                    m_react = m_react_gen(get_generator(has_randomizer<P>{}, *this), common::tagged_tuple_t<>{});
                    P::node::round(m_cur);
                } else P::node::update();
            }

            //! @brief Performs computations at round start with current time `t`.
            void round_start(times_t t) {
                P::node::round_start(t);
                maybe_align_inplace(m_neigh, has_calculus<P>{});
            }

            //! @brief Receives an incoming message (possibly reading values from sensors).
            template <typename S, typename T>
            void receive(times_t t, device_t d, common::tagged_tuple<S,T> const& m) {
                P::node::receive(t, d, m);
                fcpp::details::self(m_neigh, d) = t;
                if (m_react < TIME_MAX and t + m_react < m_next) {
                    next_time(t + m_react);
                    maybe_push_event(P::node::net, has_identifier<F>{});
                }
            }

            //! @brief Returns the reaction time for scheduling a round after a message arrives.
            times_t reaction_time() const {
                return m_react;
            }

            //! @brief Plans the reaction time for scheduling a round after a message arrives (`TIME_MAX` to prevent reacting).
            void reaction_time(times_t t) {
                m_react = t;
            }

            //! @brief Returns the time of the second most recent round (previous during rounds).
            times_t previous_time() const {
                return m_prev;
            }

            //! @brief Returns the time of the most recent round (current during rounds).
            times_t current_time() const {
                return m_cur;
            }

            //! @brief Returns the time of the next scheduled round.
            times_t next_time() const {
                return m_next;
            }

            //! @brief Plans the time of the next round (`TIME_MAX` to indicate no more rounds without removing the device).
            void next_time(times_t t) {
                if (t < TIME_MAX) {
                    assert(m_offs < TIME_MAX);
                    m_offs += t - (m_next < TIME_MAX ? m_next : m_cur);
                };
                m_next = t;
                m_mod_next = std::min(m_next, std::max(TIME_FAR, m_offs));
            }

            //! @brief Terminate round executions, causing the device to be removed from the network.
            void terminate() {
                m_mod_next = m_next = m_offs = TIME_MAX;
            }

            //! @brief Returns the time stamps of the most recent messages from neighbours.
            field<times_t> const& message_time() const {
                return m_neigh;
            }

            //! @brief Returns the time difference with neighbours.
            field<times_t> nbr_lag() const {
                return m_cur - m_neigh;
            }

            //! @brief Returns the warping factor applied to following schedulers.
            real_t frequency() const {
                return m_fact;
            }

            //! @brief Sets the warping factor applied to following schedulers.
            void frequency(real_t f) {
                if (m_offs < TIME_MAX) {
                    m_offs = m_cur - times_t(m_fact*(m_cur - m_offs)/f);
                    m_next = m_cur + times_t((m_next - m_cur) * m_fact / f);
                }
                m_fact = f;
            }

          private: // implementation details
            //! @brief Pushes a new event to the identifier.
            template <typename N>
            inline void maybe_push_event(N& n, std::true_type) {
                n.push_event(P::node::uid, m_next);
            }

            //! @brief Does not push a new event to the identifier.
            template <typename N>
            inline void maybe_push_event(N&, std::false_type) {}

            //! @brief Changes the domain of a field-like structure to match the domain of the neightbours ids.
            template <typename A>
            void maybe_align_inplace(field<A>& x, std::true_type) {
                align_inplace(x, std::vector<device_t>(fcpp::details::get_ids(P::node::nbr_uid())));
            }

            //! @brief Does not perform any alignment
            template <typename A>
            void maybe_align_inplace(field<A>&, std::false_type) {}

            //! @brief Returns the `randomizer` generator if available.
            template <typename N>
            inline auto& get_generator(std::true_type, N& n) {
                return n.generator();
            }

            //! @brief Returns a `crand` generator otherwise.
            template <typename N>
            inline crand get_generator(std::false_type, N&) {
                return {};
            }

            //! @brief Times of previous, current, next and modified-next planned rounds
            times_t m_prev, m_cur, m_next, m_mod_next;

            //! @brief The reaction time for scheduling a round after a message arrives.
            times_t m_react;

            //! @brief Times of neighbours.
            field<times_t> m_neigh;

            //! @brief Offset between the following schedule and actual times.
            times_t m_offs;

            //! @brief Warping factor for the following schedule.
            real_t m_fact;

            //! @brief The generator for the delay of reactive rounds after messages.
            reactive_delay_type m_react_gen;

            //! @brief The sequence generator.
            schedule_type m_schedule;
        };

        //! @brief The global part of the component.
        using net = typename P::net;
    };
};


}


}

#endif // FCPP_COMPONENT_TIMER_H_
