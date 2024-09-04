// Copyright © 2023 Giorgio Audrito. All Rights Reserved.

/**
 * @file graph_connector.hpp
 * @brief Implementation of the `graph_connector` component handling message exchanges between nodes of a graph net.
 */

#ifndef FCPP_CLOUD_GRAPH_CONNECTOR_H_
#define FCPP_CLOUD_GRAPH_CONNECTOR_H_

#include <cmath>

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lib/common/algorithm.hpp"
#include "lib/common/option.hpp"
#include "lib/common/serialize.hpp"
#include "lib/component/base.hpp"
#include "lib/data/field.hpp"
#include "lib/internal/twin.hpp"
#include "lib/option/distribution.hpp"

// Library to use mod functor
#include "lib/option/functor.hpp"

/**
 * @brief Namespace containing all the objects in the FCPP library.
 */
namespace fcpp {


// Namespace for all FCPP components.
namespace component {


// Namespace of tags to be used for initialising components.
namespace tags {
    //! @brief Declaration tag associating to a delay generator for sending messages after rounds (defaults to zero delay through \ref distribution::constant_n "distribution::constant_n<times_t, 0>").
    template <typename T>
    struct send_delay;

    //! @brief Declaration flag associating to whether message sizes should be emulated (defaults to false).
    template <bool b>
    struct message_size;

    //! @brief Declaration flag associating to whether parallelism is enabled (defaults to \ref FCPP_PARALLEL).
    template <bool b>
    struct parallel;

    //! @brief Declaration flag associating to whether the neighbour relation is symmetric (defaults to true).
    template <bool b>
    struct symmetric;

    //! @brief Declaration flag associating to whether the topology of the graph is static (for future use).
    template <bool b>
    struct static_topology;

    //! @brief Net initialisation tag associating to the number of threads that can be created.
    struct threads;

    //! @brief Tag associated to remainder functor computation
    template<typename A>
    struct node_splitting;

    //! @brief Tag associated to the number of MPI processes
    struct MPI_procs;
}

/**
 * @brief Component handling message exchanges between nodes of a graph net.
 *
 * If a \ref randomizer parent component is not found, \ref crand is used as random generator.
 * Any \ref simulated_connector component cannot be a parent of a \ref timer otherwise round planning may block message exchange.
 *
 * <b>Declaration tags:</b>
 * - \ref tags::send_delay defines the delay generator for sending messages after rounds (defaults to zero delay through \ref distribution::constant_n "distribution::constant_n<times_t, 0>").
 * - \ref tags::dimension defines the dimensionality of the space (defaults to 2).
 *
 * <b>Declaration flags:</b>
 * - \ref tags::message_size defines whether message sizes should be emulated (defaults to false).
 * - \ref tags::parallel defines whether parallelism is enabled (defaults to \ref FCPP_PARALLEL).
 * - \ref tags::symmetric defines whether the neighbour relation is symmetric (defaults to true).
 */
template <class... Ts>
struct graph_connector {
    //! @brief Whether message sizes should be emulated.
    constexpr static bool message_size = common::option_flag<tags::message_size, false, Ts...>;

    //! @brief Whether parallelism is enabled.
    constexpr static bool parallel = common::option_flag<tags::parallel, FCPP_PARALLEL, Ts...>;

    //! @brief Whether the neighbour relation is symmetric (defaults to true).
    constexpr static bool symmetric = common::option_flag<tags::symmetric, true, Ts...>;

    //! @brief Delay generator for sending messages after rounds.
    using delay_type = common::option_type<tags::send_delay, distribution::constant_n<times_t, 0>, Ts...>;

    //! @brief The type of settings data regulating connection.
    using connection_data_type = common::tagged_tuple_t<>;

    //! @brief The type of node splitting functor
    using node_splitting_type = common::option_type<tags::node_splitting, functor::mod<real_t, real_t>, Ts...>;

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
        DECLARE_COMPONENT(connector);
        REQUIRE_COMPONENT(connector,identifier);
        CHECK_COMPONENT(randomizer);
        //! @endcond

        //! @brief Node wrapper that can distinguish between local and remote node
        class node_accessor {
            public:
                //! @brief Constructor that initializes the node reference
                node_accessor(typename F::node * newNode){
                    node_ref = newNode;
                }

                //! @brief Message receipt method that handles the distinction between local and remote neighbour
                void receive(times_t timestamp, device_t sender_uid, typename F::node::message_t newMsg){
                    // Retrives net reference to access its methods
                    typename F::net & net_ref = node_ref->net;

                    // Computes the associated MPI process rank for both sender and receiver
                    int sender_rank = net_ref.compute_rank(sender_uid);

                    int receiver_rank = net_ref.compute_rank(node_ref->uid);

                    // check whether nodes are handled by the same process
                    if (sender_rank == receiver_rank){
                        node_ref->receive(timestamp, sender_uid, newMsg);
                    } else {
                        // receiver is a remote node: messages are added to communication map
                        net_ref.add_to_map(receiver_rank, node_ref->uid, timestamp, newMsg);
                    }
                }

                //! @brief Get method to retrieve node reference
                typename F::node * get_node_ref(){
                    return node_ref;
                }
            private:
                typename F::node * node_ref;
        };

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
            node(typename F::net& n, common::tagged_tuple<S,T> const& t) : P::node(n,t), m_delay(get_generator(has_randomizer<P>{}, *this),t), m_send(TIME_MAX), m_nbr_msg_size(0) {}

            //! @brief Destructor ensuring deadlock-free mutual disconnection.
            ~node() {
                while (m_neighbours.first().size() > 0) {
                    if (P::node::mutex.try_lock()) {
                        if (m_neighbours.first().size() > 0) {
                            typename F::node* n = (m_neighbours.first().begin()->second).get_node_ref();
                            if (n->mutex.try_lock()) {
                                m_neighbours.first().erase(m_neighbours.first().begin());
                                n->m_neighbours.second().erase(P::node::uid);
                                n->mutex.unlock();
                            }
                        }
                        P::node::mutex.unlock();
                    }
                }
                if (symmetric) return;
                while (m_neighbours.second().size() > 0) {
                    if (P::node::mutex.try_lock()) {
                        if (m_neighbours.second().size() > 0) {
                            typename F::node* n = (m_neighbours.second().begin()->second).get_node_ref();
                            if (n->mutex.try_lock()) {
                                m_neighbours.second().erase(m_neighbours.second().begin());
                                n->m_neighbours.first().erase(P::node::uid);
                                n->mutex.unlock();
                            }
                        }
                        P::node::mutex.unlock();
                    }
                }
            }

            //! @brief Adds given device to neighbours (returns true on succeed).
            bool connect(device_t i) {
                if (P::node::uid == i or m_neighbours.first().count(i) > 0) return false;
                typename F::node* n = const_cast<typename F::node*>(&P::node::net.node_at(i));
                m_neighbours.first().emplace(n->uid, node_accessor(n));
                common::unlock_guard<parallel> u(P::node::mutex);
                common::lock_guard<parallel> l(n->mutex);
                n->m_neighbours.second().emplace(P::node::uid, node_accessor(&P::node::as_final()));
                return true;
            }

            //! @brief Removes given device from neighbours (returns true on succeed).
            bool disconnect(device_t i) {
                if (P::node::uid == i or m_neighbours.first().count(i) == 0) return false;
                typename F::node* n = m_neighbours.first().at(i);
                m_neighbours.first().erase(i);
                common::unlock_guard<parallel> u(P::node::mutex);
                common::lock_guard<parallel> l(n->mutex);
                n->m_neighbours.second().erase(P::node::uid);
                return true;
            }

            //! @brief Disconnects from every neighbour (should only be used on all neighbours at once).
            void global_disconnect() {
                return;
                m_neighbours.first().clear();
                if (not symmetric) m_neighbours.second().clear();
            }

            //! @brief Checks whether a given device identifier is within neighbours.
            bool connected(device_t i) const {
                return m_neighbours.first().count(i);
            }

            //! @brief Connector data.
            connection_data_type& connector_data() {
                return m_data;
            }

            //! @brief Connector data (const access).
            connection_data_type const& connector_data() const {
                return m_data;
            }

            //! @brief Returns the time of the next sending of messages.
            times_t send_time() const {
                return m_send;
            }

            //! @brief Plans the time of the next sending of messages (`TIME_MAX` to prevent sending).
            void send_time(times_t t) {
                m_send = t;
            }

            //! @brief Disable the next sending of messages (shorthand to `send_time(TIME_MAX)`).
            void disable_send() {
                m_send = TIME_MAX;
            }

            //! @brief Size of last message sent.
            size_t msg_size() const {
                return fcpp::details::self(m_nbr_msg_size.front(), P::node::uid);
            }

            //! @brief Sizes of messages received from neighbours.
            field<size_t> const& nbr_msg_size() const {
                return m_nbr_msg_size.front();
            }

            /**
             * @brief Returns next event to schedule for the node component.
             *
             * Should correspond to the next time also during updates.
             */
            times_t next() const {
                return std::min(m_send, P::node::next());
            }

            //! @brief Updates the internal status of node component.
            void update() {
                times_t t = m_send;
                times_t pt = P::node::next();
                if (t < pt) {
                    PROFILE_COUNT("graph_connector");
                    PROFILE_COUNT("graph_connector/send");
                    m_send = TIME_MAX;
                    typename F::node::message_t m;
                    P::node::as_final().send(t, m);
                    P::node::as_final().receive(t, P::node::uid, m);
                    common::unlock_guard<parallel> u(P::node::mutex);
                    for (std::pair<device_t, node_accessor> p : m_neighbours.first()) {
                        node_accessor n = p.second;
                        // TODO: questo va lasciato???
                        common::lock_guard<parallel> l((n.get_node_ref())->mutex);
                        n.receive(t, P::node::uid, m);
                    }
                } else P::node::update();
            }


            //! @brief Performs computations at round start with current time `t`.
            void round_start(times_t t) {
                m_send = t + m_delay(get_generator(has_randomizer<P>{}, *this), common::tagged_tuple_t<>{});
                P::node::round_start(t);
            }

            //! @brief Receives an incoming message (possibly reading values from sensors).
            template <typename S, typename T>
            inline void receive(times_t t, device_t d, common::tagged_tuple<S,T> const& m) {
                P::node::receive(t, d, m);
                receive_size(common::number_sequence<message_size>{}, d, m);
            }

          private: // implementation details
            //! @brief Stores the list of neighbours in the graph.
            using neighbour_list = std::unordered_map<device_t, node_accessor>;

            //! @brief Stores size of received message (disabled).
            template <typename S, typename T>
            void receive_size(common::number_sequence<false>, device_t, common::tagged_tuple<S,T> const&) {}
            //! @brief Stores size of received message.
            template <typename S, typename T>
            void receive_size(common::number_sequence<true>, device_t d, common::tagged_tuple<S,T> const& m) {
                common::osstream os;
                os << m;
                fcpp::details::self(m_nbr_msg_size.front(), d) = os.size();
            }

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

            //! @brief A list of neighbours.
            internal::twin<neighbour_list, symmetric> m_neighbours;

            //! @brief A generator for delays in sending messages.
            delay_type m_delay;

            //! @brief Time of the next send-message event.
            times_t m_send;

            //! @brief Sizes of messages received from neighbours.
            common::option<field<size_t>, message_size> m_nbr_msg_size;

            //! @brief Data regulating the connection.
            connection_data_type m_data;
        };

        //! @brief The global part of the component.
        class net : public P::net {
            public: // visible by node objects and the main program
                //! @brief Constructor from a tagged tuple.
                template <typename S, typename T>
                explicit net(common::tagged_tuple<S,T> const& t) : 
                    P::net(t), 
                    m_threads(common::get_or<tags::threads>(t, FCPP_THREADS)),
                    m_MPI_procs_count(common::get_or<tags::MPI_procs>(t, 0)),
                    node_splitter(get_generator(has_randomizer<P>{}, *this), t) {}

                //! @brief Destructor ensuring that nodes are deleted first.
                ~net() {
                    auto n_beg = P::net::node_begin();
                    auto n_end = P::net::node_end();
                    common::parallel_for(common::tags::general_execution<parallel>(m_threads), n_end-n_beg, [&] (size_t i, size_t) {
                        n_beg[i].second.global_disconnect();
                    });

                    if (!m_communication_maps.size())
                        std::cout << "No messages exchanged" << std::endl;
                    else {
                        /*
                        for (std::pair<const int, std::unordered_map<device_t, std::pair<times_t, typename F::node::message_t>>> messages_map : m_communication_maps){
                            for (std::pair<const device_t, std::pair<times_t, typename F::node::message_t>> msg : messages_map){
                                std::cout << "Message sent to node: " + std::to_string(msg.first) + " On process :" + std::to_string(messages_map.first) + " at time: " + std::to_string(msg.second.first) << std::endl;
                            }
                        }*/
                        std::cout << "Messages have been exchanged" << std::endl;
                    }
                }

                int compute_rank(device_t target_uid){
                    /*
                    return node_splitter(
                        get_generator(has_randomizer<P>{}, *this),
                        common::make_tagged_tuple<tags::uid, tags::MPI_procs>(target_uid, m_MPI_procs_count)
                    );*/
                    return 0;
                }

                void add_to_map(int rank, device_t receiver_uid, times_t timestamp, typename F::node::message_t msg){
                    m_communication_maps[rank][receiver_uid] = std::make_pair(timestamp, msg);
                }

            //! @brief The number of threads to be used.
            size_t const m_threads;

            //! @brief Map that stores messages to be sent to remote nodes, located on another process
            std::unordered_map<int, std::unordered_map<device_t, std::pair<times_t, typename F::node::message_t>>> m_communication_maps;

            //! @brief Number of MPI processes
            size_t m_MPI_procs_count;

            //! @brief Functor to compute the MPI process associated to a node
            node_splitting_type node_splitter;

            private: // implementation details
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

                //! @brief Deletes all nodes if parent identifier.
                template <typename N>
                inline void maybe_clear(std::true_type, N& n) {
                    return n.node_clear();
                }
        };
    };
};


}


}

#endif // FCPP_CLOUD_GRAPH_CONNECTOR_H_
