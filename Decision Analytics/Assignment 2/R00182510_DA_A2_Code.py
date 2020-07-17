from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np


def Task1():
    # Subtask A:Load the input data
    data = pd.read_excel("Assignment_DA_2_a_data.xlsx", sheet_name=None)
    supp_stock = data["Supplier stock"]
    raw_mat_cost = data["Raw material costs"]
    raw_mat_ship = data["Raw material shipping"]
    prod_req = data["Product requirements"]
    prod_capacity = data["Production capacity"]
    prod_cost = data["Production cost"]
    cust_demand = data["Customer demand"]
    ship_cost = data["Shipping costs"]

    supp_stock.rename(columns={'Unnamed: 0': 'Supplier'}, inplace=True)
    supp_stock.set_index('Supplier', inplace=True)
    supp_stock.fillna(0, inplace=True)

    raw_mat_cost.rename(columns={'Unnamed: 0': 'Supplier'}, inplace=True)
    raw_mat_cost.set_index('Supplier', inplace=True)
    raw_mat_cost.fillna(0, inplace=True)

    raw_mat_ship.rename(columns={'Unnamed: 0': 'Supplier'}, inplace=True)
    raw_mat_ship.set_index('Supplier', inplace=True)
    raw_mat_ship.fillna(0, inplace=True)

    prod_req.rename(columns={'Unnamed: 0': 'Product'}, inplace=True)
    prod_req.set_index('Product', inplace=True)
    prod_req.fillna(0, inplace=True)

    prod_capacity.rename(columns={'Unnamed: 0': 'Product'}, inplace=True)
    prod_capacity.set_index('Product', inplace=True)
    prod_capacity.fillna(0, inplace=True)

    prod_cost.rename(columns={'Unnamed: 0': 'Product'}, inplace=True)
    prod_cost.set_index('Product', inplace=True)
    prod_cost.fillna(0, inplace=True)

    cust_demand.rename(columns={'Unnamed: 0': 'Product'}, inplace=True)
    cust_demand.set_index('Product', inplace=True)
    cust_demand.fillna(0, inplace=True)

    ship_cost.rename(columns={'Unnamed: 0': 'Factory'}, inplace=True)
    ship_cost.set_index('Factory', inplace=True)
    ship_cost.fillna(0, inplace=True)

    suppliers = set(supp_stock.index)
    products = set(prod_req.index)
    factories = set(ship_cost.index)
    materials = set(supp_stock.columns)
    customers = set(cust_demand.columns)

    solver = pywraplp.Solver('LPWrapper', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Subtask B: create the decision variables for the orders from the suppliers
    supplier_orders = {}
    for supplier in suppliers:
        for material in materials:
            for factory in factories:
                if supp_stock.at[supplier, material] != 0:
                    for product in products:
                        if prod_capacity.at[product, factory] != 0 and prod_req.at[product, material] != 0:
                            supplier_orders[(supplier, material, factory)] = solver.NumVar(0, supp_stock.at[
                                             supplier, material], supplier + "_" + material + "_" + factory)

    # Subtask B: create the decision variables for the production volume
    prod_volume = {}
    for product in products:
        for factory in factories:
            if prod_capacity.at[product, factory] != 0:
                prod_volume[(product, factory)] = solver.NumVar(0, prod_capacity.at[product, factory],
                                                                product + "_" + factory)

    # Subtask B: create the decision variables for the delivery to the customers
    cust_delivery = {}
    for factory in factories:
        for product in products:
            for customer in customers:
                if cust_demand.at[product, customer] != 0 and prod_capacity.at[product, factory] != 0:
                    cust_delivery[(factory, product, customer)] = solver.NumVar(0, prod_capacity.at[product, factory],
                                                                                factory + "_" + product + "_" + customer)

    # Subtask C: factories produce more than they ship to the customers
    for product in products:
        c = solver.Constraint(0, solver.infinity())
        for factory in factories:
            if prod_capacity.at[product, factory] != 0:
                c.SetCoefficient(prod_volume[(product, factory)], 1)
                for customer in customers:
                    if cust_demand.at[product, customer] != 0 and prod_capacity.at[product, factory] != 0:
                        c.SetCoefficient(cust_delivery[(factory, product, customer)], -1)

    # Subtask D: customer demand is met
    for customer in customers:
        for product in products:
            if cust_demand.at[product, customer] != 0:
                c = solver.Constraint(cust_demand.at[product, customer], cust_demand.at[product, customer])
                for factory in factories:
                    if cust_demand.at[product, customer] != 0 and prod_capacity.at[product, factory] != 0:
                        c.SetCoefficient(cust_delivery[(factory, product, customer)], 1)

    # Subtask E: suppliers have all ordered items in stock
    for material in materials:
        for supplier in suppliers:
            if supp_stock.at[supplier, material] != 0:
                c = solver.Constraint(0, supp_stock.at[supplier, material])
                for factory in factories:
                    if (supplier, material, factory) in supplier_orders.keys():
                        c.SetCoefficient(supplier_orders[(supplier, material, factory)], 1)

    #Subtask F: factories order enough materials to be able to manufacture all items
    for material in materials:
        for factory in factories:
            c = solver.Constraint(0, 0)
            for product in products:
                if prod_req.at[product, material] != 0 and (product, factory) in prod_volume.keys():
                    c.SetCoefficient(prod_volume[(product, factory)], -prod_req.at[product, material])
                    for supplier in suppliers:
                        if (supplier, material, factory) in supplier_orders.keys():
                            c.SetCoefficient(supplier_orders[(supplier, material, factory)], 1)

    # Subtask G: manufacturing capacities are not exceeded
    for product in products:
        for factory in factories:
            if prod_capacity.at[product, factory] != 0:
                c = solver.Constraint(0, prod_capacity.at[product, factory])
                c.SetCoefficient(prod_volume[(product, factory)], 1)

    # Subtask H: Cost function
    cost = solver.Objective()
    for factory in factories:
        for material in materials:
            for supplier in suppliers:
                if (supplier, material, factory) in supplier_orders.keys():
                    cost.SetCoefficient(supplier_orders[(supplier, material, factory)],
                                        raw_mat_cost.at[supplier, material])
                    cost.SetCoefficient(supplier_orders[(supplier, material, factory)],
                                        float(raw_mat_ship.at[supplier, factory]))

    for factory in factories:
        for product in products:
            if prod_capacity.at[product, factory] != 0:
                cost.SetCoefficient(prod_volume[(product, factory)], prod_cost.at[product, factory])
                for customer in customers:
                    if (factory, product, customer) in cust_delivery.keys():
                        cost.SetCoefficient(cust_delivery[(factory, product, customer)],
                                            float(ship_cost.at[factory, customer]))

    # Subtask I: Solve the linear program and determine the optimal overall cost
    cost.SetMinimization()
    status = solver.Solve()

    if status == 0:
        print("Optimal solution found")

    # Subtask J: for each factory, material has to be ordered from each individual supplier
    print('\nSupplier orders')
    for supplier in suppliers:
        for material in materials:
            for factory in factories:
                if (supplier, material, factory) in supplier_orders.keys():
                    if supplier_orders[(supplier, material, factory)].solution_value() > 0:
                        print("Order from ", supplier, " to ", factory, " is ",
                              supplier_orders[(supplier, material, factory)].solution_value(), material)

    # Subtask K: Supplier bill comprising material cost and delivery
    bill_per_supplier_factory = dict()

    for supplier in suppliers:
        bill_per_supplier_factory.__setitem__(supplier, dict())
        for factory in factories:
            material_cost = 0
            delivery_cost = 0
            for material in materials:
                if (supplier, material, factory) in supplier_orders.keys():
                    if supplier_orders[(supplier, material, factory)].solution_value() > 0:
                        material_cost += supplier_orders[(supplier, material, factory)].solution_value() * float(
                            raw_mat_cost.at[supplier, material])
                        delivery_cost += supplier_orders[(supplier, material, factory)].solution_value() * float(
                            raw_mat_ship.at[supplier, factory])
            bill_per_supplier_factory[supplier].__setitem__(factory, material_cost + delivery_cost)

    for supplier in suppliers:
        for factory in factories:
            if bill_per_supplier_factory[supplier][factory] > 0:
                print("\n", supplier, " bill for ", factory, ": ", bill_per_supplier_factory[supplier][factory])

    # Subtask L: Units of products manufactured by each factory and manufacturing cost
    print('\nProducts produced by each factory')
    cost_per_factory = dict.fromkeys(factories, 0)
    for factory in factories:
        total_manufact_cost = 0
        for product in products:
            if (product, factory) in prod_volume.keys():
                if prod_volume[(product, factory)].solution_value() > 0:
                    #                     print(prod_volume[(product, factory)].solution_value())
                    print(round(prod_volume[(product, factory)].solution_value()), " ", product, "are produced by",
                          factory)
                    total_manufact_cost += prod_volume[(product, factory)].solution_value() * prod_cost.at[
                        product, factory]
        cost_per_factory[factory] = total_manufact_cost

    for factory in cost_per_factory.keys():
        print('\nTotal manufacturing cost for ', factory, ": ", cost_per_factory[factory])

    # Subtask M: Products shipped by each factory to each customer and the total shipping cost per customer
    print('\nProducts delivered to each customer')
    cost_per_customer = dict.fromkeys(customers, 0)
    for customer in customers:
        total_ship_cost = 0
        for product in products:
            for factory in factories:
                if (factory, product, customer) in cust_delivery.keys():
                    if cust_delivery[(factory, product, customer)].solution_value() > 0:
                        print(cust_delivery[(factory, product, customer)].solution_value(), product,
                              " items delivered to ", customer, " from ", factory)
                        total_ship_cost += cust_delivery[(factory, product, customer)].solution_value() * ship_cost.at[
                            factory, customer]
        cost_per_customer[customer] = total_ship_cost

    for customer in cost_per_customer.keys():
        print("\nTotal shipping cost for ", customer, ": ", cost_per_customer[customer])
    print('\n')

    for factory in factories:
        for customer in customers:
            for product in products:
                for material in materials:
                    if prod_req.at[product, material] != 0 and (factory, product, customer) in cust_delivery.keys():
                        if cust_delivery[(factory, product, customer)].solution_value() > 0:
                            materials_req = cust_delivery[(factory, product, customer)].solution_value() * prod_req.at[
                                product, material]

                            print(materials_req, material, " required for making", product, ' to ', customer, "from",
                                  factory)


def Task2():
    #Subtask A: Load input data
    data = pd.read_excel("Assignment_DA_2_b_data.xlsx", sheet_name=None)
    flight_sched = data["Flight schedule"]
    taxi_dist = data["Taxi distances"]
    terminal_capacity = data["Terminal capacity"]

    flight_sched.rename(columns={'Unnamed: 0': 'Flight'}, inplace=True)
    flight_sched.set_index('Flight', inplace=True)
    flight_sched.fillna(0, inplace=True)
    taxi_dist.rename(columns={'Unnamed: 0': 'Runway'}, inplace=True)
    taxi_dist.set_index('Runway', inplace=True)
    taxi_dist.fillna(0, inplace=True)
    terminal_capacity.rename(columns={'Unnamed: 0': 'Terminal'}, inplace=True)
    terminal_capacity.set_index('Terminal', inplace=True)
    terminal_capacity.fillna(0, inplace=True)

    print(flight_sched, '\n\n', taxi_dist, '\n\n', terminal_capacity)

    flights = list(flight_sched.index)
    runways = list(taxi_dist.index)
    terminals = list(terminal_capacity.index)

    solver = pywraplp.Solver('LPWrapper', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # Subtask B. Decision variables for arrival runway allocation, departure runway allocation and for the terminal allocation
    arrival_runway = {}
    departure_runway = {}
    for flight in flights:
        for runway in runways:
            arrival_runway[(flight, runway)] = solver.BoolVar("arrival_%s_%s" % (flight, runway))
            departure_runway[(flight, runway)] = solver.BoolVar("departure_%s_%s" % (flight, runway))

    terminal_alloc = {}
    for flight in flights:
        for terminal in terminals:
            terminal_alloc[(flight, terminal)] = solver.BoolVar("terminal_%s_%s" % (flight, terminal))

    # Subtask C. Auxillary Variables for the taxi movements between runways and terminals for each flight
    arrival_runway_to_terminal = {}
    for terminal in terminals:
        for runway in runways:
            for  flight in flights:
                arrival_runway_to_terminal[(flight, terminal, runway)] = solver.BoolVar(
                    "arrival_%s_%s_%s" % (flight, terminal, runway))

    departure_runway_to_terminal = {}
    for terminal in terminals:
        for runway in runways:
            for flight in flights:
                departure_runway_to_terminal[(flight, terminal, runway)] = solver.BoolVar(
                    "departure_%s_%s_%s" % (flight, terminal, runway))

    for terminal in terminals:
        for runway in runways:
            total_flights_using_terminal_runway = solver.IntVar(0, solver.infinity(),
                                                                "flight_count" + terminal + "_" + runway)

    # Subtask D. Every flight has exactly two taxi movements
    for terminal in terminals:
        for runway in runways:
            c = solver.Constraint(1, 1)
            for flight in flights:
                c.SetCoefficient(departure_runway_to_terminal[(flight, terminal, runway)], 1)

    for terminal in terminals:
        for runway in runways:
            c = solver.Constraint(1, 1)
            for flight in flights:
                c.SetCoefficient(arrival_runway_to_terminal[(flight, terminal, runway)], 1)

    # Subtask E. taxi movements of a flight are to and from the allocated terminal
    # and
    # Subtask F. taxi movements of a flight include the allocated arrival and departure runways
    for terminal in terminals:
        for runway in runways:
            for flight in flights:
                solver.Add(arrival_runway_to_terminal[(flight, terminal, runway)] >=
                           (arrival_runway[(flight, runway)] + terminal_alloc[(flight, terminal)]) - 1)
                solver.Add(arrival_runway_to_terminal[(flight, terminal, runway)] <= arrival_runway[(flight, runway)])
                solver.Add(arrival_runway_to_terminal[(flight, terminal, runway)] <= terminal_alloc[(flight, terminal)])

                solver.Add(departure_runway_to_terminal[(flight, terminal, runway)] >=
                           (departure_runway[(flight, runway)] + terminal_alloc[(flight, terminal)]) - 1)
                solver.Add(
                    departure_runway_to_terminal[(flight, terminal, runway)] <= departure_runway[(flight, runway)])
                solver.Add(
                    departure_runway_to_terminal[(flight, terminal, runway)] <= terminal_alloc[(flight, terminal)])

    # Subtask G. Each flight has exactly one allocated arrival runway and one allocated departure runway
    for flight in flights:
        c = solver.Constraint(1, 1)
        for runway in runways:
            c.SetCoefficient(arrival_runway[(flight, runway)], 1)

    for flight in flights:
        c = solver.Constraint(1, 1)
        for runway in runways:
            c.SetCoefficient(departure_runway[(flight, runway)], 1)

    # Subtask H. Each flight is allocated to exactly one terminal
    for flight in flights:
        c = solver.Constraint(1, 1)
        for terminal in terminal:
            c.SetCoefficient(departure_runway[(flight, runway)], 1)

    # Subtask I. No runway is used by more than one flight during each timeslot
    for runway in runways:
        for flight in flights:
            variables = []
            other_flights = [value for value in flights if value != flight]
            for other in other_flights:
                if flight_sched.at[flight, 'Arrival'] == flight_sched.at[other, 'Arrival']:
                    variables.append(arrival_runway[(flight, runway)])
                    variables.append(arrival_runway[(other, runway)])
                if flight_sched.at[flight, 'Arrival'] == flight_sched.at[other, 'Departure']:
                    variables.append(arrival_runway[(flight, runway)])
                    variables.append(departure_runway[(other, runway)])
                if flight_sched.at[flight, 'Departure'] == flight_sched.at[other, 'Arrival']:
                    variables.append(departure_runway[(flight, runway)])
                    variables.append(arrival_runway[(other, runway)])
                if flight_sched.at[flight, 'Departure'] == flight_sched.at[other, 'Departure']:
                    variables.append(departure_runway[(flight, runway)])
                    variables.append(departure_runway[(other, runway)])
            solver.Add(sum(variables) <= 1)

    # Subtask J. Terminal capacities are not exceeded
    for terminal in terminals:
        solver.Add(
            sum([terminal_alloc[(flight, terminal)] for flight in flights]) <= terminal_capacity.at[terminal, 'Gates'])

    for terminal in terminals:
        for runway in runways:
            c = solver.Constraint(0, 0)
            for flight in flights:
                c.SetCoefficient(total_flights_using_terminal_runway, 1)
                c.SetCoefficient(arrival_runway_to_terminal[(flight, terminal, runway)], -1)
                c.SetCoefficient(departure_runway_to_terminal[(flight, terminal, runway)], -1)

    # Subtask K: implement the objective function
    cost = solver.Objective()
    for terminal in terminals:
        for runway in runways:
            cost.SetCoefficient(total_flights_using_terminal_runway, float(taxi_dist.at[runway, terminal]))

    status = solver.Solve()

    print("\nStatus:", status)


def main():
    # Execute Task_1 or Task_2 by specifying the exact task name as a value to the execute variable
    execute = 'Task_1'

    if execute == 'Task_1':
        Task1()
    elif execute == 'Task_2':
        Task2()
    else:
        print("The task to be executed is invalid")

main()