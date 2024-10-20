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
#include <mpi.h>

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

    //! @brief Tag associated to graph partitioning functor
    template<typename A>
    struct node_splitting;

    //! @brief Tag associated to the number of MPI processes
    struct mpi_procs{};

    //! @brief Node initialisation tag associating to a `device_t` unique identifier.
    struct uid;

    //! @brief Period between successive invocations of the node update method
    template <typename T>
    struct MPI_schedule{};
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
    using node_splitting_type = common::option_type<tags::node_splitting, functor::mod<tags::uid, tags::mpi_procs, size_t>, Ts...>;

    //! @brief Type of delay between successive invocation of the update method
    using schedule_type = common::option_type<tags::MPI_schedule, distribution::constant_n<times_t, 0>, Ts...>;

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
                node_accessor(device_t uid){
                    ref_uid = uid;
                }

                //! @brief Message receipt method that handles the distinction between local and remote neighbour
                void receive(typename F::net & loc_net_ref, times_t timestamp, device_t sender_uid, typename F::node::message_t newMsg){
                    // Retrives net reference to access its methods

                    // Computes the associated MPI process rank for both sender and receiver
                    int sender_rank = loc_net_ref.compute_rank(sender_uid);
                    int receiver_rank = loc_net_ref.compute_rank(ref_uid);

                    // check whether nodes are handled by the same process
                    if (sender_rank == receiver_rank){
                        // Retriving the reference to the neighbour via uid
                        typename F::node* n = const_cast<typename F::node*>(&loc_net_ref.node_at(ref_uid));
                        common::lock_guard<parallel> l(n->mutex);
                        n->receive(timestamp, sender_uid, newMsg);
                    } else {
                        // receiver is a remote node: messages are added to communication map
                        loc_net_ref.add_to_map(receiver_rank, ref_uid, timestamp, newMsg);
                    }
                }

                //! @brief Get method to retrieve node reference
                device_t get_node_ref(){
                    return ref_uid;
                }
            private:
                device_t ref_uid;
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
            node(typename F::net& n, common::tagged_tuple<S,T> const& t) : P::node(n,t), m_delay(get_generator(has_randomizer<P>{}, *this),t), m_send(TIME_MAX), m_nbr_msg_size(0) {
            }

            //! @brief Destructor ensuring deadlock-free mutual disconnection.
            ~node() {
                int sender_rank = P::node::net.compute_rank(P::node::uid);
                int receiver_rank;

                while (m_neighbours.first().size() > 0) {
                    if (P::node::mutex.try_lock()) {
                        if (m_neighbours.first().size() > 0) {
                            device_t node_uid = (m_neighbours.first().begin()->second).get_node_ref();
                            receiver_rank = P::node::net.compute_rank(node_uid);
                            // if the two nodes are on the same MPI process, a physical arc must be removed
                            if (sender_rank == receiver_rank){
                                typename F::node* n = const_cast<typename F::node*>(&P::node::net.node_at(node_uid));
                                if (n->mutex.try_lock()) {
                                    n->m_neighbours.second().erase(P::node::uid);
                                    n->mutex.unlock();
                                }
                            }
                            m_neighbours.first().erase(m_neighbours.first().begin());
                        }
                        P::node::mutex.unlock();
                    }
                }
                if (symmetric) return;
                while (m_neighbours.second().size() > 0) {
                    if (P::node::mutex.try_lock()) {
                        if (m_neighbours.second().size() > 0) {
                            device_t node_uid = (m_neighbours.second().begin()->second).get_node_ref();
                            receiver_rank = P::node::net.compute_rank(node_uid);
                            if (sender_rank == receiver_rank){
                                typename F::node* n = const_cast<typename F::node*>(&P::node::net.node_at(node_uid));
                                if (n->mutex.try_lock()) {
                                    n->m_neighbours.first().erase(P::node::uid);
                                    n->mutex.unlock();
                                }
                            }
                            m_neighbours.second().erase(m_neighbours.second().begin());
                        }
                        P::node::mutex.unlock();
                    }
                }
            }

            //! @brief Adds given device to neighbours (returns true on succeed).
            bool connect(device_t i) {
                if (P::node::uid == i or m_neighbours.first().count(i) > 0) return false;
                
                // Attualmente questa istruzione non funziona, perchè in questo punto net
                // non è ancora inizializzato
                //int sender_rank = P::node::net.compute_rank(P::node::uid);
                int sender_rank = P::node::uid % 2;
                //int receiver_rank = P::node::net.compute_rank(i);                
                int receiver_rank = i % 2;                

                m_neighbours.first().emplace(i, node_accessor(i));
                common::unlock_guard<parallel> u(P::node::mutex);
                if (sender_rank == receiver_rank){
                    // local neigbour: setting up a standard connection
                    typename F::node* n = const_cast<typename F::node*>(&P::node::net.node_at(i));
                    common::lock_guard<parallel> l(n->mutex);
                    n->m_neighbours.second().emplace(P::node::uid, node_accessor(P::node::uid));
                }
                
                return true;
            }

            //! @brief Removes given device from neighbours (returns true on succeed).
            bool disconnect(device_t i) {
                int sender_rank = P::node::uid % 2;
                //int sender_rank = P::node::net.compute_rank(P::node::uid);
                int receiver_rank = i % 2;  
                //int receiver_rank = P::node::net.compute_rank(i);   

                if (P::node::uid == i or m_neighbours.first().count(i) == 0) return false;

                m_neighbours.first().erase(i);
                common::unlock_guard<parallel> u(P::node::mutex);
                
                if (sender_rank == receiver_rank){
                    // physical connection: there is another arc that must be removed
                    // in a deadlock-free way
                    typename F::node* n = m_neighbours.first().at(i);
                    common::lock_guard<parallel> l(n->mutex);
                    n->m_neighbours.second().erase(P::node::uid);
                }
                
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
                        // non serve copiare p.second
                        node_accessor n = p.second;
                        n.receive(P::node::net, t, P::node::uid, m);
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
                    m_send(TIME_MAX),
                    m_threads(common::get_or<tags::threads>(t, FCPP_THREADS)),
                    m_MPI_procs_count(common::get<tags::mpi_procs>(t)),
                    node_splitter(get_generator(has_randomizer<P>{}, *this), t){}

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
                        std::cout << "Messages have been exchanged" << std::endl;
                        for (std::pair<const int, std::unordered_map<device_t, std::tuple<times_t, typename F::node::message_t, device_t>>> messages_map : m_communication_maps){
                            for (std::pair<const device_t, std::tuple<times_t, typename F::node::message_t, device_t>> msg : messages_map.second){
                                std::cout << "Message sent to node: " + std::to_string(msg.first) + " On process: " + std::to_string(messages_map.first) + " at time: " + std::to_string(std::get<0>(msg.second)) << std::endl;
                            }
                        }
                    }

                    // releasing the lock that protects the remote messages map
                    common::unlock_guard<parallel> u(comm_map_mutex);
                }

                int compute_rank(device_t target_uid){
                    return node_splitter(
                        get_generator(has_randomizer<P>{}, *this),
                        common::make_tagged_tuple<tags::uid, tags::mpi_procs>(target_uid, m_MPI_procs_count)
                    );
                }

                void add_to_map(int rank, device_t receiver_uid, times_t timestamp, typename F::node::message_t msg, device_t sender_uid){
                    common::lock_guard<parallel> l(comm_map_mutex);
                    m_communication_maps[rank][receiver_uid] = std::make_tuple(timestamp, msg, sender_uid);
                }

                //! @brief Updates the internal status of net component.
                void update() {
                    times_t t = m_send;
                    times_t pt = P::net::next();
                    // verifico quale evento di update vada eseguito prima
                    // tra il mio e quello del padre
                    if (t < pt) {
                        /*
                            1) Individuare lista messaggi da inviare
                            2) capire come funzioni l'invio
                                -che metodo si usa?
                                    -forse asincrono, per non dover aspettare la receive degli altri
                                    -forse bloccante, ma solo nel senso di ripartire con l'esecuzione
                                    quando sono sicuro che il messaggio sia in viaggio
                                -chi è il destinatario?
                                    -io mando il messaggio al processo MPI, il codice di invio e ricezione
                                    viene eseguito da qualsiasi nodo (perchè in effetti tutti hanno
                                    la loro mappa con i messaggi da inviare)
                                    -l'idea è che il primo che riceve i messaggi si occupa di smistarli, quindi
                                    non serve eleggere un leader. L'unico potenziale problema è evitare
                                    la duplicazione dei messaggi (questo si risolve verificando che gli accesssi
                                    al buffer del processo mpi siano sincronizzati)
                                -serve serializzare i messaggi da inviare?
                            3) capire come fare la ricezione
                                -che metodo si usa?
                                -cosa ricevo?
                                -chi riceve?
                                -serve lo smistamento?
                        */
                        // TODO: ha senso? Cosa fa?
                        PROFILE_COUNT("graph_connector");
                        PROFILE_COUNT("graph_connector/send");

                        // sending messages to remote nodes
                        common::osstream os;
                        int snd_buffer_size = 0;
                        for (std::pair<const int, std::unordered_map<device_t, std::tuple<times_t, typename F::node::message_t, device_t>>> messages_map : m_communication_maps){
                            std::cerr << "Sending remote message to process: " << messages_map.first << std::endl;
                            os << messages_map.second;
                            snd_buffer_size = os.size();
                            std::vector<char> m_data = std::move(os.data());
                            // 0 for a size message, 1 for a data message
                            MPI_Send(&snd_buffer_size, 1, MPI_INT, messages_map.first, 0, MPI_COMM_WORLD);
                            // move dovrebbe evitare di fare una copia dei dati, operazione molto costosa
                            MPI_Send(&m_data[0], snd_buffer_size, MPI_CHAR, messages_map.first, 1, MPI_COMM_WORLD);
                        }

                        // receiving messages from remote node using MPI_receive
                        int rcv_buffer_size;
                        // Ha senso fare una receive sola, contando che potrebbero essersi accumulati
                        // i dati provenienti da tanti processi MPI diversi??
                        // non bisognerebbe fare un polling di tutti i processi MPI??
                        int messageExists = 0;
                        for (int rank = 0; rank < m_MPI_procs_count - 1; rank++){
                            messageExists = 0;
                            std::cerr << "Checking remote message from process: " << rank << std::endl;
                            rcv_buffer_size = 0;
                            // TODO: rendere non bloccante qualora non ci siano messaggi dal
                            // processo di rango rank
                            MPI_Iprobe(rank, 0, MPI_COMM_WORLD, &messageExists, MPI_STATUS_IGNORE);
                            if (messageExists){
                                std::cerr << "Receiving remote message from process: " << rank << std::endl;
                                MPI_Recv(&rcv_buffer_size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                // TODO: si può rendere più efficiente evitando di riallocare ogni volta?
                                std::vector<char>rcv_buffer(rcv_buffer_size);
                                // qui bisogna fare un loop per considerare ogni rango
                                // per ciascuno, assicurarsi che la computazione non si blocchi se non ci sono messaggi
                                MPI_Recv(&rcv_buffer[0], rcv_buffer_size, MPI_CHAR, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                //common::isstream is(std::move(rcv_buffer));
                                common::isstream is(std::move(rcv_buffer));
                                std::unordered_map<device_t, std::tuple<times_t, typename F::node::message_t, device_t>> incoming_msg_map;
                                is >> incoming_msg_map;

                                // ora scansiono la mappa, smistando i messaggi ai destinatari
                                for (std::pair<const device_t, std::tuple<times_t, typename F::node::message_t, device_t>> msg : incoming_msg_map){
                                    std::cerr << "Sending message to local node " << msg.first << std::endl;
                                    // recupero il puntatore a nodo
                                    typename F::node* n = const_cast<typename F::node*>(&P::net.node_at(msg.first));
                                    // e lo uso per richiamare la receive
                                    common::lock_guard<parallel> l(n->mutex);
                                    n->receive(std::get<0>(msg.second), std::get<2>(msg.second), std::get<1>(msg.second));
                                }
                            } else
                                std::cerr << "No message coming from process of rank: " << rank << std::endl;
                        }
                    } 
                        else P::net::update();
                }

                //! @brief Performs computations at round start with current time `t`.
                void round_start(times_t t) {
                    m_send = t + m_delay(get_generator(has_randomizer<P>{}, *this), common::tagged_tuple_t<>{});
                    P::node::round_start(t);
                }

                /**
                * @brief Returns next event to schedule for the node component.
                *
                * Should correspond to the next time also during updates.
                */
                times_t next() const {
                    return std::min(m_send, P::net::next());
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

                //! @brief The number of threads to be used.
                size_t const m_threads;

                //! @brief Map that stores messages to be sent to remote nodes, located on another process
                std::unordered_map<int, std::unordered_map<device_t, std::tuple<times_t, typename F::node::message_t, device_t>>> m_communication_maps;

                //! @brief Number of MPI processes
                int m_MPI_procs_count;

                //! @brief Functor to compute the MPI process associated to a node
                node_splitting_type node_splitter;

                //! @brief Mutex to manage parallel access to messages map
                common::mutex<parallel> comm_map_mutex;

                //! @brief Time of the next send-message event.
                times_t m_send;

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
