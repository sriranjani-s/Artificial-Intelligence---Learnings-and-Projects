from ortools.sat.python import cp_model

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables_ = variables

    def OnSolutionCallback(self):    
        print("Next solution:")
        for variable in self.variables_:
            print("  ", self.Value(variable))
        
def main():
    model = cp_model.CpModel()
    
    x1 = model.NewIntVar(0,10,'x1')
    x2 = model.NewIntVar(0,10,'x2')

    model.Add(x1 + x2 < 10)
    
    product =  model.NewIntVar(0,100,'pr')
    model.AddProdEquality(product, [x1,x2])   
    model.Add(product > 15)
       
    print("Search for all solutions")
    solver = cp_model.CpSolver()    
    solver.SearchForAllSolutions(model, SolutionPrinter([x1,x2]))

    
    print()
    print("Search for an optimal solution")

    model.Maximize(x2 - x1)

    status = solver.Solve(model)    
    print(solver.StatusName(status))

    print("x1 = ", solver.Value(x1))
    print("x2 = ", solver.Value(x2))
    
main()
