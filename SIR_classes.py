# Standard Library
import math
import os
import random
from io import TextIOWrapper
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Third Party Library
import japanize_matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pams.agents.base import Agent
from pams.agents.fcn_agent import FCNAgent
from pams.agents.high_frequency_agent import HighFrequencyAgent
from pams.logs.base import CancelLog, ExecutionLog, Logger, OrderLog
from pams.logs.market_step_loggers import MarketStepSaver
from pams.market import Market
from pams.order import LIMIT_ORDER, MARKET_ORDER, Cancel, Order
from pams.runners import SequentialRunner
from pams.session import Session
from pams.simulator import Simulator
from pams.utils.json_random import JsonRandom
from scipy import stats
from scipy.stats import linregress

MARGIN_FIXED = 0
MARGIN_NORMAL = 1


class SIRNoiseAgent(FCNAgent):
    """SIR Agent class Noise ver.

    This class inherits from the :class:`pams.agents.FCNAgent` class.

    When the agent is susceptible, it submits an empty order.
    When the agent is infected, it submits an order whose expected return
    has only noise_log_return.
    """

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """agent setup.  Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration.
                                       This must include the parameters "numInitialInfection",
                                       "network", "lambda", and "myu".
            accessible_markets_ids (List[int]): list of market IDs.

        Returns:
            None
        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        json_random: JsonRandom = JsonRandom(prng=self.prng)

        self.infection_state: str = "susceptible"

    def get_outstanding_orders(self, market: Market, agent_id) -> List[Order]:
        """get the list of unexecuted buy orders.

        Returns:
            List[Order]: the list of unexecuted buy orders
        """
        outstanding_orders = []

        for order in market.buy_order_book.priority_queue:
            if order.agent_id == agent_id:
                outstanding_orders.append(order)

        return outstanding_orders

    def submit_orders(
        self,
        markets: List[Market],
        *args: Any,
    ) -> List[Union[Order, Cancel]]:
        """submit orders based on FCNI-based calculation.
        If the agent is susceptible, it submits an empty order.
        If the agent is infected, it submits a buy order whose expected return has only noise_log_return.
        If the agent is infected, it submits a sell order whose expected return has only noise_log_return.

        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        if markets[0].get_time() <= 100:
            orders = []

            return orders

        if self.infection_state in ["susceptible", "recovered-2"]:
            orders = []

            return orders

        elif self.infection_state == "infected":
            orders = super().submit_orders(markets)

            for i, order in enumerate(orders):
                order.is_buy = True
                order.ttl = 10000

            return orders

        else:  # recovered-1
            orders = super().submit_orders(markets)

            for i, order in enumerate(orders):
                order.is_buy = False
                order.volume = (
                    1 if self.asset_volumes[0] == 0 else self.asset_volumes[0]
                )
                order.ttl = 10000

            outstanding_orders = self.get_outstanding_orders(
                market=markets[0], agent_id=self.agent_id
            )
            for order in outstanding_orders:
                cancel_order = Cancel(order=order)
                orders.append(cancel_order)

            return orders

    def submit_orders_by_market(self, market: Market) -> List[Union[Order, Cancel]]:
        """submit orders by market (internal usage).
        Noise log return is only used to the expected return of the market order.

        Args:
            market (Market): market to order.

        Returns:
            List[Union[Order, Cancel]]: order list.
        """
        orders: List[Union[Order, Cancel]] = []
        if not self.is_market_accessible(market_id=market.market_id):
            return orders

        time: int = market.get_time()
        time_window_size: int = min(time, self.time_window_size)
        assert time_window_size >= 0
        assert self.fundamental_weight >= 0.0
        assert self.chart_weight >= 0.0
        assert self.noise_weight >= 0.0

        fundamental_scale: float = 1.0 / max(self.mean_reversion_time, 1)
        fundamental_log_return = fundamental_scale * math.log(
            market.get_fundamental_price() / market.get_market_price()
        )
        assert self.is_finite(fundamental_log_return)

        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_mean_log_return = chart_scale * math.log(
            market.get_market_price() / market.get_market_price(time - time_window_size)
        )
        assert self.is_finite(chart_mean_log_return)

        noise_log_return: float = self.noise_scale * self.prng.gauss(mu=0.0, sigma=1.0)
        assert self.is_finite(noise_log_return)

        expected_log_return: float = noise_log_return

        assert self.is_finite(expected_log_return)

        expected_future_price: float = market.get_market_price() * math.exp(
            expected_log_return * self.time_window_size
        )
        assert self.is_finite(expected_future_price)

        if self.margin_type == MARGIN_FIXED:
            assert 0.0 <= self.order_margin <= 1.0

            order_volume: int = 1

            if expected_future_price > market.get_market_price():
                order_price = expected_future_price * (1 - self.order_margin)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
            if expected_future_price < market.get_market_price():
                order_price = expected_future_price * (1 + self.order_margin)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )

        if self.margin_type == MARGIN_NORMAL:
            assert self.order_margin >= 0.0
            order_price = (
                expected_future_price
                + self.prng.gauss(mu=0.0, sigma=1.0) * self.order_margin
            )
            order_volume = 1
            assert order_price >= 0.0
            assert order_volume > 0
            if expected_future_price > market.get_market_price():
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
            if expected_future_price < market.get_market_price():
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )

        return orders


class SIRSequentialRunner(SequentialRunner):
    """Even if SIR agent submits an empty order, session is finished."""

    def _setup(self) -> None:
        """runner setup. (Internal method)"""
        super()._setup()
        i = 0
        self.market_price_doubled = False
        self.SIR_agents_list = []
        self.FCN_agents_list = []
        self.SIR_agents_id_list = []
        self.FCN_agents_id_list = []

        for agent in self.simulator.agents:
            if hasattr(agent, "infection_state"):
                self.SIR_agents_list.append(
                    agent
                )  ############これ使えばもうちょい他も綺麗に描けそう
                self.SIR_agents_id_list.append(agent.agent_id)

            else:
                self.FCN_agents_list.append(agent)
                self.FCN_agents_id_list.append(agent.agent_id)
        self.network_type = self.settings["SIRAgents"]["networkType"]
        self.network = self.create_network(self.network_type)
        self.SIRneighbors_id: dict = self.get_SIRneighbors_id()
        self.lam: float = self.settings["SIRAgents"]["lambda"]
        self.myu: float = self.settings["SIRAgents"]["myu"]

        self.S_sum_list: List = []
        self.I_sum_list: List = []
        self.R_sum_list: List = []
        self.S_sum: int = (
            self.settings["SIRAgents"]["numAgents"]
            - self.settings["SIRAgents"]["numInitialInfections"]
        )
        self.I_sum: int = self.settings["SIRAgents"]["numInitialInfections"]
        self.R_sum: int = 0

        self.buy_num = 0
        self.sell_num = 0

        self.SIR_buy_order_vol = {}
        self.SIR_sell_order_vol = {}
        self.FCN_buy_order_vol = {}
        self.FCN_sell_order_vol = {}

        self.SIR_buy_execution = {}
        self.FCN_buy_execution = {}
        self.SIR_sell_execution = {}
        self.FCN_sell_execution = {}

        self.submit_order_logs: List[Dict] = []
        self.execution_logs: List[Dict] = []

        self.network_centrality: Dict = self.calculate_centrality_to_dict(self.network)

        self.initial_infected_id = []

        # agent is infected based on centrality
        m = self.settings["SIRAgents"]["numInitialInfections"]
        sorted_centrality = sorted(
            self.network_centrality.items(),
            key=lambda item: item[1]["degree centrality"],
            reverse=True,
        )
        top_m_agents = [item[0] for item in sorted_centrality[:m]]

        for agent in self.SIR_agents_list:
            if agent.agent_id in top_m_agents:
                agent.infection_state = "infected"
                self.initial_infected_id.append(agent.agent_id)

        # infection state is added to self.network_centrality dict
        for agent in self.SIR_agents_list:
            agent_id = agent.agent_id
            infection_state = agent.infection_state

            if agent_id in self.network_centrality:
                self.network_centrality[agent_id][
                    "initial infection state"
                ] = infection_state
            else:
                self.network_centrality[agent_id] = {
                    "initial infection state": infection_state
                }

        for agent in self.SIR_agents_list:
            agent_id = agent.agent_id

            # calculate distance to initial infected
            min_distance = float("inf")
            for infected_id in self.initial_infected_id:
                try:
                    distance = nx.shortest_path_length(
                        self.network, source=agent_id, target=infected_id
                    )
                    min_distance = min(min_distance, distance)
                except nx.NetworkXNoPath:
                    pass

            # distace is added to self.network_centrality dict
            if agent_id in self.network_centrality:
                self.network_centrality[agent_id][
                    "distance to initial infected"
                ] = min_distance
            else:
                self.network_centrality[agent_id] = {
                    "distance to initial infected": min_distance
                }

    def create_network(self, network_type):
        """
        Generate various types of networks.

        Args:
            network_type (str): The type of network to generate ('BA', 'ER', 'WS', or 'complete').


        Returns:
            networkx.Graph: The generated network.

        Notes:
            n: total number of nodes
            k: first neighbors of each nodes (for WS network)
            p: probability of rewiring (for WS network)
            m: number of edges to which the new node will connect (for BA network)
        """
        num_agents = self.settings["SIRAgents"]["numAgents"]

        n = num_agents
        k = 4
        p = 0.2
        m = 2

        if network_type == "BA":
            if n == 10:
                network = nx.barabasi_albert_graph(n, m + 1)

            else:
                network = nx.barabasi_albert_graph(n, m)
        elif network_type == "ER":
            network = nx.erdos_renyi_graph(n, k / n)
        elif network_type == "WS":
            network = nx.watts_strogatz_graph(n, k, p)
        elif network_type == "complete":
            network = nx.complete_graph(num_agents)
        else:
            raise ValueError("unknown network type: " + network_type)

        # mapping of node ID
        if len(self.SIR_agents_id_list) == num_agents:
            mapping = {i: self.SIR_agents_id_list[i] for i in range(num_agents)}
            network = nx.relabel_nodes(network, mapping)
        else:
            raise ValueError(
                "The length of SIR_agents_id_list does not match num_agents."
            )

        return network

    def get_network(self):
        """
        Get the network which is used in this simulation.

        Args:
            None

        Returns:
            network
        """
        return self.network

    def calculate_centrality_to_dict(self, network):
        """
        Calculate various centralities for a network and store them in a dictionary.

        Args:
            network (networkx.Graph): The network to calculate centralities for.

        Returns:
            list[dict]: various centralities of each agents.
        """

        degree_centrality = nx.degree_centrality(network)
        betweenness_centrality = nx.betweenness_centrality(network)
        closeness_centrality = nx.closeness_centrality(network)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(network, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            print("Warning: eigenvector_centrality did not converge.")
            eigenvector_centrality = {node: 0 for node in network.nodes()}

        centrality_dict = {}
        for node in network.nodes():
            centrality_dict[node] = {
                "degree centrality": degree_centrality[node],
                "betweenness centrality": betweenness_centrality[node],
                "closeness centrality": closeness_centrality[node],
                "eigenvector centrality": eigenvector_centrality[node],
            }

        return centrality_dict

    def get_SIRneighbors_id(self) -> Dict:
        SIRneighbors_id = {}
        for agent in self.simulator.agents:
            if hasattr(agent, "infection_state"):
                neighbors_agent_ids = list(self.network.neighbors(agent.agent_id))
                SIRneighbors_id[agent.agent_id] = neighbors_agent_ids

        return SIRneighbors_id

    def _collect_orders_from_normal_agents(
        self, session: Session
    ) -> List[List[Union[Order, Cancel]]]:
        """collect orders from normal_agents. (Internal method)
        orders are corrected until the total number of orders reaches max_normal_orders

        Args:
            session (Session): session.

        Returns:
            List[List[Union[Order, Cancel]]]: orders lists.
        """
        agents = self.simulator.normal_frequency_agents
        agents = self._prng.sample(agents, len(agents))
        n_orders = 0
        all_orders: List[List[Union[Order, Cancel]]] = []

        for agent in agents:
            if n_orders >= session.max_normal_orders:
                break
            n_orders += 1

            if session.session_id == 1:
                if hasattr(agent, "infection_state"):
                    agent.infection_state = self.update_infection_state(agent)

                self.S_sum_list.append(self.S_sum)
                self.I_sum_list.append(self.I_sum)
                self.R_sum_list.append(self.R_sum)

            orders = agent.submit_orders(markets=self.simulator.markets)
            if session.session_id == 1:
                if hasattr(agent, "infection_state") and orders:
                    if orders[0].is_buy:
                        self.SIR_buy_order_vol[self.simulator.markets[0].get_time()] = (
                            self.SIR_buy_order_vol.get(
                                self.simulator.markets[0].get_time(), 0
                            )
                            + orders[0].volume
                        )
                    else:
                        self.SIR_sell_order_vol[
                            self.simulator.markets[0].get_time()
                        ] = (
                            self.SIR_buy_order_vol.get(
                                self.simulator.markets[0].get_time(), 0
                            )
                            - orders[0].volume
                        )
                if not hasattr(agent, "infection_state"):
                    if orders[0].is_buy:
                        self.FCN_buy_order_vol[self.simulator.markets[0].get_time()] = (
                            self.FCN_buy_order_vol.get(
                                self.simulator.markets[0].get_time(), 0
                            )
                            + orders[0].volume
                        )
                    else:
                        self.FCN_sell_order_vol[
                            self.simulator.markets[0].get_time()
                        ] = (
                            self.FCN_buy_order_vol.get(
                                self.simulator.markets[0].get_time(), 0
                            )
                            - orders[0].volume
                        )

                if orders:
                    network_centrality = (
                        self.network_centrality[agent.agent_id]
                        if agent in self.SIR_agents_list
                        else {}
                    )
                    self.submit_order_logs.append(
                        {
                            "market time": self.simulator.markets[0].get_time(),
                            "market price": self.simulator.markets[
                                0
                            ].get_market_price(),
                            "best buy price": self.simulator.markets[
                                0
                            ].get_best_buy_price(),
                            "best sell price": self.simulator.markets[
                                0
                            ].get_best_sell_price(),
                            "order type": "buy" if orders[0].is_buy else "sell",
                            "order price": orders[0].price,
                            "order volume": orders[0].volume,
                            "agent type": (
                                agent.infection_state
                                if agent in self.SIR_agents_list
                                else "FCN"
                            ),
                            "agent id": agent.agent_id,
                            "time window size": agent.time_window_size,
                            **(
                                {"fundamental weight": agent.fundamental_weight}
                                if agent in self.FCN_agents_list
                                else {}
                            ),
                            **(
                                {"chart weight": agent.chart_weight}
                                if agent in self.FCN_agents_list
                                else {}
                            ),
                            **(
                                {"noise weight": agent.noise_weight}
                                if agent in self.FCN_agents_list
                                else {}
                            ),
                            **network_centrality,
                        }
                    )

                else:
                    pass

            if len(orders) > 0:
                if not session.with_order_placement:
                    raise AssertionError("currently order is not accepted")
                if sum([order.agent_id != agent.agent_id for order in orders]) > 0:
                    raise ValueError(
                        "spoofing order is not allowed. please check agent_id in order"
                    )

                all_orders.append(orders)

        return all_orders

    def update_infection_state(self, agent) -> str:
        """update infection state.

        Args:None

        Returns:
            agent.infection_state (str): infection state.
        """
        ratio_I_neighbors, ratio_R_neighbors = self.get_neighbors_summary(agent)
        if (
            self.simulator.markets[0].get_market_price()
            >= 2 * self.simulator.markets[0].get_fundamental_price()
        ):
            self.market_price_doubled = True

        if agent.infection_state == "susceptible":
            if self._prng.random() < ratio_I_neighbors * self.lam:
                self.S_sum -= 1
                self.I_sum += 1

                return "infected"

            else:
                return "susceptible"

        elif agent.infection_state == "infected":
            if self.market_price_doubled:
                if agent.agent_id in self.initial_infected_id:
                    self.R_sum += 1
                    self.I_sum -= 1

                    return "recovered-1"

                if self._prng.random() < ratio_R_neighbors * self.myu:
                    self.R_sum += 1
                    self.I_sum -= 1

                    return "recovered-1"

                else:
                    return "infected"
            else:
                return "infected"

        else:
            return "recovered-2"

    def get_neighbors_summary(self, agent):
        """Count the number of infected and recovered neighbors and calculate their ratios.

        Args:
            agent (Agent): The agent for which to calculate neighbor summary.

        Returns:
            float: Ratio of infected neighbors to total neighbors.
            float: Ratio of recovered neighbors to total neighbors.
        """
        num_I_neighbors = 0
        num_R_neighbors = 0
        total_neighbors = len(self.SIRneighbors_id[agent.agent_id])

        for neighbor_id in self.SIRneighbors_id[agent.agent_id]:
            neighbor = self.simulator.agents[neighbor_id]
            if neighbor.infection_state == "infected":
                num_I_neighbors += 1
            elif neighbor.infection_state in ["recovered-1", "recovered-2"]:
                num_R_neighbors += 1

        if total_neighbors > 0:
            ratio_I_neighbors = num_I_neighbors / total_neighbors
            ratio_R_neighbors = num_R_neighbors / total_neighbors
        else:
            ratio_I_neighbors = 0
            ratio_R_neighbors = 0

        return ratio_I_neighbors, ratio_R_neighbors

    def get_SIR_PL_dataset(self):
        """make SIR agents' PL dataset.

        Args:
            None

        Returns:
            SIR_PL_dataset (pd.dataset):
        """
        before_pl = (
            self.settings["SIRAgents"]["assetVolume"]
            * self.settings["Market"]["marketPrice"]
            + self.settings["SIRAgents"]["cashAmount"]
        )
        pl_list = []
        assetvolume_list = []
        cash_list = []
        degree_centrality_list = []
        betweenness_centrality_list = []
        closeness_centrality_list = []
        eigenvector_centrality_list = []
        distance_initial_infected_list = []

        for agent in self.SIR_agents_list:
            pl = (
                agent.asset_volumes[0] * self.simulator.markets[0].get_market_price()
                + agent.cash_amount
            ) - before_pl
            pl_list.append(pl)
            assetvolume_list.append(agent.asset_volumes[0])
            cash_list.append(agent.cash_amount)

            centrality_data = self.network_centrality.get(agent.agent_id, {})
            degree_centrality_list.append(
                centrality_data.get("degree centrality", None)
            )
            betweenness_centrality_list.append(
                centrality_data.get("betweenness centrality", None)
            )
            closeness_centrality_list.append(
                centrality_data.get("closeness centrality", None)
            )
            eigenvector_centrality_list.append(
                centrality_data.get("eigenvector centrality", None)
            )
            distance_initial_infected_list.append(
                centrality_data.get("distance to initial infected", None)
            )

        SIR_PL_dataset = pd.DataFrame(
            {
                "agent id": self.SIR_agents_id_list,
                "# of SIR": self.settings["SIRAgents"]["numAgents"],
                "AssetVolume": assetvolume_list,
                "Cash": cash_list,
                "PL": pl_list,
                "Degree Centrality": degree_centrality_list,
                "Betweenness Centrality": betweenness_centrality_list,
                "Closeness Centrality": closeness_centrality_list,
                "Eigenvector Centrality": eigenvector_centrality_list,
                "Distance to initial infected": distance_initial_infected_list,
            }
        )

        return SIR_PL_dataset

    def get_FCN_PL_dataset(self):
        """count the number of infected neighbors.

        Args:
            None

        Returns:
            num_I_neighbors (int): the number of infected neighbors.
        """
        """make SIR agents' PL dataset.

        Args:
            None

        Returns:
            SIR_PL_dataset (pd.dataset): 
        """
        before_pl = (
            self.settings["FCNAgents"]["assetVolume"]
            * self.settings["Market"]["marketPrice"]
            + self.settings["FCNAgents"]["cashAmount"]
        )
        pl_list = []
        assetvolume_list = []
        cash_list = []
        wf_list = []
        wc_list = []
        wn_list = []
        rt_list = []
        tw_list = []

        for agent in self.FCN_agents_list:
            pl = (
                agent.asset_volumes[0] * self.simulator.markets[0].get_market_price()
                + agent.cash_amount
            ) - before_pl
            pl_list.append(pl)
            assetvolume_list.append(agent.asset_volumes[0])
            cash_list.append(agent.cash_amount)
            wf_list.append(agent.fundamental_weight)
            wc_list.append(agent.chart_weight)
            wn_list.append(agent.noise_weight)
            rt_list.append(agent.mean_reversion_time)
            tw_list.append(agent.time_window_size)

        FCN_PL_dataset = pd.DataFrame(
            {
                "agent id": self.FCN_agents_id_list,
                "# of SIR": self.settings["SIRAgents"]["numAgents"],
                "fundamental weight": wf_list,
                "chart weight": wc_list,
                "Noise weight": wn_list,
                "mean reversion time": rt_list,
                "time window size": tw_list,
                "AssetVolume": assetvolume_list,
                "Cash": cash_list,
                "PL": pl_list,
            }
        )

        return FCN_PL_dataset

    def _handle_orders(
        self, session: Session, local_orders: List[List[Union[Order, Cancel]]]
    ) -> List[List[Union[Order, Cancel]]]:
        """handle orders. (Internal method)
        processing local orders and correct and process the orders from high frequency agents.

        Args:
            session (Session): session.
            local_orders (List[List[Union[Order, Cancel]]]): local orders.

        Returns:
            List[List[Union[Order, Cancel]]]: order lists.
        """
        sequential_orders = self._prng.sample(local_orders, len(local_orders))
        all_orders: List[List[Union[Order, Cancel]]] = [*sequential_orders]
        for orders in sequential_orders:
            for order in orders:
                if not session.with_order_placement:
                    raise AssertionError("currently order is not accepted")
                market: Market = self.simulator.id2market[order.market_id]
                if isinstance(order, Order):
                    self.simulator._trigger_event_before_order(order=order)
                    log: OrderLog = market._add_order(order=order)
                    agent: Agent = self.simulator.id2agent[order.agent_id]
                    agent.submitted_order(log=log)
                    self.simulator._trigger_event_after_order(order_log=log)
                elif isinstance(order, Cancel):
                    self.simulator._trigger_event_before_cancel(cancel=order)
                    log_: CancelLog = market._cancel_order(cancel=order)
                    agent = self.simulator.id2agent[order.order.agent_id]
                    agent.canceled_order(log=log_)
                    self.simulator._trigger_event_after_cancel(cancel_log=log_)
                else:
                    raise NotImplementedError
                if session.with_order_execution:
                    logs: List[ExecutionLog] = market._execution()
                    self.simulator._update_agents_for_execution(execution_logs=logs)

                    for execution_log in logs:
                        agent = self.simulator.id2agent[execution_log.buy_agent_id]
                        buy_agent = agent
                        if agent in self.SIR_agents_list:
                            self.SIR_buy_execution[execution_log.time] = (
                                self.SIR_buy_execution.get(execution_log.time, 0) + 1
                            )
                        else:
                            self.FCN_buy_execution[execution_log.time] = (
                                self.FCN_buy_execution.get(execution_log.time, 0) + 1
                            )

                        agent.executed_order(log=execution_log)
                        agent = self.simulator.id2agent[execution_log.sell_agent_id]
                        sell_agent = agent
                        if agent in self.SIR_agents_list:
                            self.SIR_sell_execution[execution_log.time] = (
                                self.SIR_sell_execution.get(execution_log.time, 0) - 1
                            )
                        else:
                            self.FCN_sell_execution[execution_log.time] = (
                                self.FCN_sell_execution.get(execution_log.time, 0) - 1
                            )

                        agent.executed_order(log=execution_log)
                        self.simulator._trigger_event_after_execution(
                            execution_log=execution_log
                        )

                        buy_centrality = {}
                        if buy_agent in self.SIR_agents_list:
                            for key, value in self.network_centrality[
                                buy_agent.agent_id
                            ].items():
                                buy_centrality[f"buy {key}"] = value

                        sell_centrality = {}
                        if sell_agent in self.SIR_agents_list:
                            for key, value in self.network_centrality[
                                sell_agent.agent_id
                            ].items():
                                sell_centrality[f"sell {key}"] = value

                        self.execution_logs.append(
                            {
                                "market time": execution_log.time,
                                "execution price": execution_log.price,
                                "execution volume": execution_log.volume,
                                "buy agent type": (
                                    buy_agent.infection_state
                                    if buy_agent in self.SIR_agents_list
                                    else "FCN"
                                ),
                                "buy agent id": buy_agent.agent_id,
                                **(
                                    {
                                        "buy fundamental weight": buy_agent.fundamental_weight
                                    }
                                    if buy_agent in self.FCN_agents_list
                                    else {}
                                ),
                                **(
                                    {"buy chart weight": buy_agent.chart_weight}
                                    if buy_agent in self.FCN_agents_list
                                    else {}
                                ),
                                **(
                                    {"buy noise weight": buy_agent.noise_weight}
                                    if buy_agent in self.FCN_agents_list
                                    else {}
                                ),
                                **buy_centrality,
                                "sell agent type": (
                                    sell_agent.infection_state
                                    if sell_agent in self.SIR_agents_list
                                    else "FCN"
                                ),
                                "sell agent id": sell_agent.agent_id,
                                **(
                                    {
                                        "sell fundamental weight": sell_agent.fundamental_weight
                                    }
                                    if sell_agent in self.FCN_agents_list
                                    else {}
                                ),
                                **(
                                    {"sell chart weight": sell_agent.chart_weight}
                                    if sell_agent in self.FCN_agents_list
                                    else {}
                                ),
                                **(
                                    {"sell noise weight": sell_agent.noise_weight}
                                    if sell_agent in self.FCN_agents_list
                                    else {}
                                ),
                                **sell_centrality,
                            }
                        )

            if session.high_frequency_submission_rate < self._prng.random():
                continue

            n_high_freq_orders = 0
            agents = self.simulator.high_frequency_agents
            agents = self._prng.sample(agents, len(agents))
            for agent in agents:
                if n_high_freq_orders >= session.max_high_frequency_orders:
                    break

                high_freq_orders: List[Union[Order, Cancel]] = agent.submit_orders(
                    markets=self.simulator.markets
                )
                if len(high_freq_orders) > 0:
                    if not session.with_order_placement:
                        raise AssertionError("currently order is not accepted")
                    if (
                        sum(
                            [
                                order.agent_id != agent.agent_id
                                for order in high_freq_orders
                            ]
                        )
                        > 0
                    ):
                        raise ValueError(
                            "spoofing order is not allowed. please check agent_id in order"
                        )
                    all_orders.append(high_freq_orders)

                    n_high_freq_orders += 1

                    for order in high_freq_orders:
                        market = self.simulator.id2market[order.market_id]
                        if isinstance(order, Order):
                            self.simulator._trigger_event_before_order(order=order)
                            log = market._add_order(order=order)
                            agent = self.simulator.id2agent[order.agent_id]
                            agent.submitted_order(log=log)
                            self.simulator._trigger_event_after_order(order_log=log)
                        elif isinstance(order, Cancel):
                            self.simulator._trigger_event_before_cancel(cancel=order)
                            log_ = market._cancel_order(cancel=order)
                            agent = self.simulator.id2agent[order.order.agent_id]
                            agent.canceled_order(log=log_)
                            self.simulator._trigger_event_after_cancel(cancel_log=log_)
                        else:
                            raise NotImplementedError
                        if session.with_order_execution:
                            logs = market._execution()
                            self.simulator._update_agents_for_execution(
                                execution_logs=logs
                            )
                            for execution_log in logs:
                                agent = self.simulator.id2agent[
                                    execution_log.buy_agent_id
                                ]
                                agent.executed_order(log=execution_log)
                                agent = self.simulator.id2agent[
                                    execution_log.sell_agent_id
                                ]
                                agent.executed_order(log=execution_log)
                                self.simulator._trigger_event_after_execution(
                                    execution_log=execution_log
                                )
        return all_orders
