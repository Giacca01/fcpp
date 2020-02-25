// Copyright © 2020 Giorgio Audrito. All Rights Reserved.

/**
 * @file identifier.hpp
 * @brief Implementation of the `identifier` component handling node creation and indexing.
 */

#ifndef FCPP_COMPONENT_IDENTIFIER_H_
#define FCPP_COMPONENT_IDENTIFIER_H_

#include <map>
#include <queue>
#include <type_traits>

#include "lib/settings.hpp"
#include "lib/common/algorithm.hpp"
#include "lib/common/mutex.hpp"
#include "lib/common/random_access_map.hpp"
#include "lib/common/tagged_tuple.hpp"
#include "lib/component/base.hpp"


/**
 * @brief Namespace containing all the objects in the FCPP library.
 */
namespace fcpp {


//! @brief Namespace for all FCPP components.
namespace component {


//! @brief Namespace of tags to be used for initialising components.
namespace tags {
    //! @brief Tag associating to the time sensitivity, allowing indeterminacy below it.
    struct epsilon {};

    //! @brief Tag associating to the number of threads that can be used.
    struct threads {};
}


/**
 * Priority queue of pairs `(times_t, device_t)` designed for popping bunches of elements at a time.
 *
 * @param synchronised Whether lots of collisions (same time) are to be expected or not.
 */
template <bool synchronised>
class times_queue;

//! @brief Specialisation for lots of collisions, as map of vectors.
template <>
class times_queue<true> {
  public:
    //! @brief Default constructor.
    times_queue() {
        m_queue.emplace(TIME_MAX, std::vector<device_t>());
    }
    
    //! @brief The smallest time in the queue.
    inline times_t next() const {
        return m_queue.begin()->first;
    }
    
    //! @brief Adds a new pair to the queue.
    inline void push(times_t t, device_t uid) {
        m_queue[t].push_back(uid);
    }

    //! @brief Pops elements with the smaller time if up to `t`.
    inline std::vector<device_t> pop(times_t t) {
        if (next() > t) return {};
        std::vector<device_t> v = std::move(m_queue.begin()->second);
        m_queue.erase(m_queue.begin());
        return v;
    }
    
  private:
    //! @brief The actual priority queue.
    std::map<times_t, std::vector<device_t>> m_queue;
};

//! @brief Specialisation for few collisions, as priority queue.
template <>
class times_queue<false> {
  public:
    //! @brief Default constructor.
    times_queue() {
        m_queue.emplace(TIME_MAX, device_t());
    }
    
    //! @brief The smallest time in the queue.
    inline times_t next() const {
        return m_queue.top().first;
    }
    
    //! @brief Adds a new pair to the queue.
    inline void push(times_t t, device_t uid) {
        m_queue.emplace(t, uid);
    }

    //! @brief Pops elements with times up to `t`.
    std::vector<device_t> pop(times_t t) {
        std::vector<device_t> v;
        while (next() <= t) {
            v.push_back(m_queue.top().second);
            m_queue.pop();
        }
        return v;
    }
    
  private:
    //! @brief The type of queue elements.
    using type = std::pair<times_t, device_t>;
    //! @brief The actual priority queue.
    std::priority_queue<type, std::vector<type>, std::greater<type>> m_queue;
};


/**
 * @brief Component handling node creation and indexing.
 *
 * Initialises `net` with tag `epsilon` associating to the time sensitivity, allowing indeterminacy below it (defaults to `FCPP_TIME_EPSILON`); and with tag `threads` associating to the number of threads that can be used (defaults to `FCPP_THREADS`).
 * Must be unique in a composition of components.
 * Optimises queuing of events according to `synchronised`.
 *
 * @param synchronised  Whether to assume that many events are going to happen at the same time.
 */
template <bool synchronised>
struct identifier {
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
        //! @brief Marks that an identifier component is present.
        struct identifier_tag {};
        
        //! @brief Checks if T has a `identifier_tag`.
        template <typename T, typename = int>
        struct has_itag : std::false_type {};
        template <typename T>
        struct has_itag<T, std::conditional_t<true,int,typename T::identifier_tag>> : std::true_type {};
        
        //! @brief Asserts that P has no `identifier_tag`.
        static_assert(not has_itag<P>::value, "cannot combine multiple identifier components");

        //! @brief The local part of the component.
        using node = typename P::node;
        
        //! @brief The global part of the component.
        class net : public P::net {
          public: // visible by node objects and the main program
            //! @brief The map type used internally for storing nodes.
            using map_type = common::random_access_map<device_t, typename F::node>;
            
            //! @brief Constructor from a tagged tuple.
            template <typename S, typename T>
            net(const common::tagged_tuple<S,T>& t) : P::net(t), m_next_uid(0), m_epsilon(common::get_or<tags::epsilon>(t, FCPP_TIME_EPSILON)), m_threads(common::get_or<tags::threads>(t, FCPP_THREADS)) {}

            /**
             * @brief Returns next event to schedule for the net component.
             *
             * Should correspond to the next time also during updates.
             */
            times_t next() const {
                return std::min(m_queue.next(), P::net::next());
            }
            
            //! @brief Updates the internal status of net component.
            void update() {
                if (m_queue.next() < P::net::next()) {
                    std::vector<device_t> nv = m_queue.pop(m_queue.next() + m_epsilon);
                    common::parallel_for(common::tags::general_execution<FCPP_PARALLEL>(m_threads), nv.size(), [&nv,this](size_t i, size_t){
                        if (m_nodes.count(nv[i]) > 0) {
                            typename F::node& n = m_nodes.at(nv[i]);
                            common::lock_guard<FCPP_PARALLEL> device_lock(n.mutex);
                            n.update();
                        }
                    });
                    for (device_t uid : nv) if (m_nodes.count(uid) > 0) m_queue.push(m_nodes.at(uid).next(), uid);
                } else P::net::update();
            }
            
            //! @brief Returns the total number of nodes.
            inline size_t node_size() const {
                return m_nodes.size();
            }
            
            //! @brief Returns whether a node with a certain device identifier exists.
            inline size_t node_count(device_t uid) const {
                return m_nodes.count(uid);
            }
            
            //! @brief Const access to the node with a given device device identifier.
            inline const typename F::node& node_at(device_t uid) const {
                return m_nodes.at(uid);
            }

            //! @brief Access to the node with a given device device identifier (given a lock for the node's mutex).
            typename F::node& node_at(device_t uid, common::unique_lock<FCPP_PARALLEL>& l) {
                l = common::unique_lock<FCPP_PARALLEL>(m_nodes.at(uid).mutex);
                return m_nodes.at(uid);
            }
            
          protected: // visible by net objects only
            //! @brief Random-access iterator to the first node (in a random order).
            typename map_type::const_iterator node_begin() const {
                return m_nodes.begin();
            }
            
            //! @brief Random-access const iterator to the first node (in a random order).
            typename map_type::iterator node_begin() {
                return m_nodes.begin();
            }
            
            //! @brief Random-access iterator to the last node (in a random order).
            typename map_type::const_iterator node_end() const {
                return m_nodes.end();
            }
            
            //! @brief Random-access const iterator to the last node (in a random order).
            typename map_type::iterator node_end() {
                return m_nodes.end();
            }
            
            //! @brief Creates a new node, initialising it with data in `t` (returns the identifier assigned).
            template <typename S, typename T>
            device_t node_emplace(const common::tagged_tuple<S,T>& t) {
                while (m_nodes.count(m_next_uid) > 0) ++m_next_uid;
                using tt_type = typename common::tagged_tuple<S,T>::template push_back<tags::uid, device_t>;
                tt_type tt(t);
                common::get<tags::uid>(tt) = m_next_uid;
                m_nodes.emplace(std::piecewise_construct, std::make_tuple(m_next_uid), std::tuple<typename F::net&, tt_type>(P::net::as_final(), tt));
                m_queue.push(m_nodes.at(m_next_uid).next(), m_next_uid);
                return m_next_uid++;
            }

            //! @brief Erases the node with a given identifier.
            inline size_t node_erase(device_t uid) {
                return m_nodes.erase(uid);
            }
            
          private: // implementation details
            //! @brief The set of nodes, indexed by identifier.
            map_type m_nodes;
            
            //! @brief The queue of identifiers by next event.
            times_queue<synchronised> m_queue;
            
            //! @brief The next free identifier.
            device_t m_next_uid;
            
            //! @brief The time sensitivity.
            const times_t m_epsilon;
            
            //! @brief The number of threads to be used.
            const size_t m_threads;
        };
    };
};


}


}

#endif // FCPP_COMPONENT_IDENTIFIER_H_