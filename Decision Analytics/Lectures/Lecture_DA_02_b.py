from ortools.linear_solver import pywraplp

def main():
    solver = pywraplp.Solver('LPWrapper', 
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    
    x = solver.NumVar(0, solver.infinity(), 'x')
    y = solver.NumVar(0, solver.infinity(), 'y')
        
    constraint1 = solver.Constraint(0, 10)
    constraint1.SetCoefficient(x, 1)
    constraint1.SetCoefficient(y, 2)

    constraint2 = solver.Constraint(0, 5)
    constraint2.SetCoefficient(x, -2)
    constraint2.SetCoefficient(y, 1)

    objective = solver.Objective()
    objective.SetCoefficient(x, 3)
    objective.SetCoefficient(y, 4)
    objective.SetMaximization()

  
    solver.Solve()
    
    print ("x = " , x.solution_value())
    print ("y = ",  y.solution_value())
    print("objective = ", 3 * x.solution_value() + 4 * y.solution_value())

main()
