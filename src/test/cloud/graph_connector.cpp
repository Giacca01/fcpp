// Copyright © 2023 Giorgio Audrito. All Rights Reserved.

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "lib/component/base.hpp"
#include "lib/component/identifier.hpp"
#include "lib/component/timer.hpp"
#include "lib/cloud/graph_connector.hpp"

#include "test/helper.hpp"

using namespace fcpp;
using namespace component::tags;


struct tag {};
struct gat {};
struct oth {};

// Component exposing the storage interface.
struct exposer {
    template <typename F, typename P>
    struct component : public P {
        using node = typename P::node;
        struct net : public P::net {
            using P::net::net;
            using P::net::node_emplace;
        };
    };
};

using seq_per = sequence::periodic<distribution::constant_n<times_t, 2>, distribution::constant_n<times_t, 1>, distribution::constant_n<times_t, 9>>;

template <int O>
using combo = component::combine_spec<
    exposer,
    component::timer<round_schedule<seq_per>>,
    component::graph_connector<message_size<(O & 2) == 2>, parallel<(O & 1) == 1>, send_delay<distribution::constant_n<times_t, 1, 4>>>,
    component::identifier<
        parallel<(O & 1) == 1>,
        synchronised<(O & 2) == 2>
    >,
    component::base<parallel<(O & 1) == 1>>
>;

MULTI_TEST(GraphConnectorTest, Arcs, O, 2) {
    typename combo<O>::net n{common::make_tagged_tuple<oth>("foo")};
    n.node_emplace(common::make_tagged_tuple<uid>(0));
    n.node_emplace(common::make_tagged_tuple<uid>(1));
    n.node_emplace(common::make_tagged_tuple<uid>(2));
    n.node_emplace(common::make_tagged_tuple<uid>(3));
    n.node_emplace(common::make_tagged_tuple<uid>(4));

    EXPECT_EQ(false, n.node_at(1).connected(0));
    {
        typename combo<O>::net::lock_type l;
        n.node_at(1, l).connect(0);
    }
    EXPECT_EQ(true, n.node_at(1).connected(0));
    {
        typename combo<O>::net::lock_type l;
        n.node_at(1, l).disconnect(0);
    }
    EXPECT_EQ(false, n.node_at(1).connected(0));
}

MULTI_TEST(GraphConnectorTest, Messages, O, 2) {
    auto update = [](auto& node) {
        common::lock_guard<(O & 1) == 1> l(node.mutex);
        node.update();
    };
    typename combo<O>::net  network{common::make_tagged_tuple<oth>("foo")};
    typename combo<O>::node d0{network, common::make_tagged_tuple<uid>(0)};
    typename combo<O>::node d1{network, common::make_tagged_tuple<uid>(1)};
    typename combo<O>::node d2{network, common::make_tagged_tuple<uid>(2)};
    typename combo<O>::node d3{network, common::make_tagged_tuple<uid>(3)};
    typename combo<O>::node d4{network, common::make_tagged_tuple<uid>(4)};
    EXPECT_EQ(2, d0.next());
    EXPECT_EQ(2, d1.next());
    EXPECT_EQ(2, d2.next());
    EXPECT_EQ(2, d3.next());
    EXPECT_EQ(2, d4.next());
    update(d0);
    update(d1);
    update(d2);
    update(d3);
    update(d4);
    EXPECT_EQ(2.25, d0.next());
    EXPECT_EQ(2.25, d1.next());
    EXPECT_EQ(2.25, d2.next());
    EXPECT_EQ(2.25, d3.next());
    EXPECT_EQ(2.25, d4.next());
    update(d0);
    update(d1);
    update(d2);
    update(d3);
    update(d4);
    EXPECT_EQ(3, d0.next());
    EXPECT_EQ(3, d1.next());
    EXPECT_EQ(3, d2.next());
    EXPECT_EQ(3, d3.next());
    EXPECT_EQ(3, d4.next());
    update(d0);
    update(d1);
    update(d2);
    update(d3);
    update(d4);
    EXPECT_EQ(3.25, d0.next());
    EXPECT_EQ(3.25, d1.next());
    EXPECT_EQ(3.25, d2.next());
    EXPECT_EQ(3.25, d3.next());
    EXPECT_EQ(3.25, d4.next());
}
