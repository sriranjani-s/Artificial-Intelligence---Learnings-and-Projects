from ortools.sat.python import cp_model

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables_ = variables
        self.solutions_ = 0

    def OnSolutionCallback(self):
        self.solutions_ = self.solutions_ + 1
        print("solution", self.solutions_ )
        i=0
        for vars_in_timestep in self.variables_:
            i=i+1
#            print(" - Timestep: ", i)
            for op in vars_in_timestep:
                if self.Value(vars_in_timestep[op]):
                    print("   ", op)
        print()
        
def main():
    model = cp_model.CpModel()

    maxT = 8
    
    WolfOnThisSide = []   
    SheepOnThisSide = []
    CabbageOnThisSide = []
    FerrymanOnThisSide = []
    
    WolfOnOppositeSide = []
    SheepOnOppositeSide = []
    CabbageOnOppositeSide = []
    FerrymanOnOppositeSide = []
    
    for t in range(maxT):    
        WolfOnThisSide.append(model.NewBoolVar("WolfOnThisSide"+str(t)))   
        SheepOnThisSide.append(model.NewBoolVar("SheepOnThisSide"+str(t)))    
        CabbageOnThisSide.append(model.NewBoolVar("CabbageOnThisSide"+str(t)))
        FerrymanOnThisSide.append(model.NewBoolVar("FerrymanOnThisSide"+str(t)))
        
        WolfOnOppositeSide.append(model.NewBoolVar("WolfOnThisSide"+str(t)))    
        SheepOnOppositeSide.append(model.NewBoolVar("SheepOnThisSide"+str(t)))    
        CabbageOnOppositeSide.append(model.NewBoolVar("CabbageOnThisSide"+str(t)))
        FerrymanOnOppositeSide.append(model.NewBoolVar("FerrymanOnOppositeSide"+str(t)))
    
    # Initial state
    model.AddBoolAnd( 
            [FerrymanOnThisSide[0], 
             WolfOnThisSide[0], 
             SheepOnThisSide[0], 
             CabbageOnThisSide[0] ] )
    model.AddBoolAnd( 
            [FerrymanOnOppositeSide[0].Not(), 
             WolfOnOppositeSide[0].Not(), 
             SheepOnOppositeSide[0].Not(), 
             CabbageOnOppositeSide[0].Not() ] )
    
    # Goal state
    model.AddBoolAnd(
            [WolfOnOppositeSide[maxT-1], 
             SheepOnOppositeSide[maxT-1], 
             CabbageOnOppositeSide[maxT-1]])

    # Operator encodings
    moveWolfAccross = []
    moveSheepAccross = []
    moveCabbageAccross = []
    moveWolfBack = []
    moveSheepBack = []
    moveCabbageBack = []
    moveAccross = []
    moveBack = []
    for t in range(maxT-1):
        moveWolfAccross.append(model.NewBoolVar("MoveWolfAccross"+str(t)))
        moveSheepAccross.append(model.NewBoolVar("MoveSheepAccross"+str(t)))
        moveCabbageAccross.append(model.NewBoolVar("MoveCabbageAccross"+str(t)))
        moveWolfBack.append(model.NewBoolVar("MoveWolfBack"+str(t)))
        moveSheepBack.append(model.NewBoolVar("MoveSheepBack"+str(t)))
        moveCabbageBack.append(model.NewBoolVar("MoveCabbageBack"+str(t)))
        moveAccross.append(model.NewBoolVar("MoveAccross"+str(t)))
        moveBack.append(model.NewBoolVar("MoveBack"+str(t)))
                   
        model.AddBoolAnd([WolfOnThisSide[t],FerrymanOnThisSide[t],
                          WolfOnOppositeSide[t+1], FerrymanOnOppositeSide[t+1], 
                          WolfOnThisSide[t+1].Not(), FerrymanOnThisSide[t+1].Not()
                          ]).OnlyEnforceIf(moveWolfAccross[t])
        model.AddBoolAnd([WolfOnOppositeSide[t], FerrymanOnOppositeSide[t],
                          WolfOnThisSide[t+1], FerrymanOnThisSide[t+1], 
                          WolfOnOppositeSide[t+1].Not(), FerrymanOnOppositeSide[t+1].Not()
                          ]).OnlyEnforceIf(moveWolfBack[t])
        
        model.AddBoolAnd([SheepOnThisSide[t], FerrymanOnThisSide[t],
                          SheepOnOppositeSide[t+1], FerrymanOnOppositeSide[t+1], SheepOnThisSide[t+1].Not(), FerrymanOnThisSide[t+1].Not()
                          ]).OnlyEnforceIf(moveSheepAccross[t])
        model.AddBoolAnd([SheepOnOppositeSide[t], FerrymanOnOppositeSide[t],
                          SheepOnThisSide[t+1], FerrymanOnThisSide[t+1], SheepOnOppositeSide[t+1].Not(), FerrymanOnOppositeSide[t+1].Not()
                          ]).OnlyEnforceIf(moveSheepBack[t])

        model.AddBoolAnd([CabbageOnThisSide[t], FerrymanOnThisSide[t],
                          CabbageOnOppositeSide[t+1], FerrymanOnOppositeSide[t+1], CabbageOnThisSide[t+1].Not(), FerrymanOnThisSide[t+1].Not()
                          ]).OnlyEnforceIf(moveCabbageAccross[t])
        model.AddBoolAnd([CabbageOnOppositeSide[t], FerrymanOnOppositeSide[t],
                          CabbageOnThisSide[t+1], FerrymanOnThisSide[t+1], CabbageOnOppositeSide[t+1].Not(), FerrymanOnOppositeSide[t+1].Not()
                          ]).OnlyEnforceIf(moveCabbageBack[t])

        model.AddBoolAnd([FerrymanOnThisSide[t],
                          FerrymanOnOppositeSide[t+1], FerrymanOnThisSide[t+1].Not()
                          ]).OnlyEnforceIf(moveAccross[t])
        model.AddBoolAnd([FerrymanOnOppositeSide[t], 
                          FerrymanOnThisSide[t+1], FerrymanOnOppositeSide[t+1].Not()
                          ]).OnlyEnforceIf(moveBack[t])
    
    # Frame axioms (no state is switched on without an action)
    for t in range(maxT-1):
        model.AddBoolOr([WolfOnThisSide[t+1].Not(), WolfOnThisSide[t], moveWolfBack[t]])
        model.AddBoolOr([WolfOnOppositeSide[t+1].Not(), WolfOnOppositeSide[t], moveWolfAccross[t]])
        model.AddBoolOr([SheepOnThisSide[t+1].Not(), SheepOnThisSide[t], moveSheepBack[t]])
        model.AddBoolOr([SheepOnOppositeSide[t+1].Not(), SheepOnOppositeSide[t], moveSheepAccross[t]])
        model.AddBoolOr([CabbageOnThisSide[t+1].Not(), CabbageOnThisSide[t], moveCabbageBack[t]])
        model.AddBoolOr([CabbageOnOppositeSide[t+1].Not(), CabbageOnOppositeSide[t], moveCabbageAccross[t]])
        model.AddBoolOr([FerrymanOnThisSide[t+1].Not(), 
                         FerrymanOnThisSide[t], 
                         moveWolfBack[t], 
                         moveSheepBack[t], 
                         moveCabbageBack[t], 
                         moveBack[t]])
        model.AddBoolOr([FerrymanOnOppositeSide[t+1].Not(), 
                         FerrymanOnOppositeSide[t], 
                         moveWolfAccross[t], 
                         moveSheepAccross[t], 
                         moveCabbageAccross[t], 
                         moveAccross[t]])
    
    # Complete exclusion axiom (only one action at a time)
    for t in range(maxT-1):
        model.AddBoolOr([moveWolfAccross[t].Not(), moveSheepAccross[t].Not()])        
        model.AddBoolOr([moveWolfAccross[t].Not(), moveCabbageAccross[t].Not()])
        model.AddBoolOr([moveWolfAccross[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveWolfAccross[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveWolfAccross[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveWolfAccross[t].Not(), moveAccross[t].Not()])
        model.AddBoolOr([moveWolfAccross[t].Not(), moveBack[t].Not()])
        model.AddBoolOr([moveSheepAccross[t].Not(), moveCabbageAccross[t].Not()])
        model.AddBoolOr([moveSheepAccross[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveSheepAccross[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveSheepAccross[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveSheepAccross[t].Not(), moveAccross[t].Not()])
        model.AddBoolOr([moveSheepAccross[t].Not(), moveBack[t].Not()])
        model.AddBoolOr([moveCabbageAccross[t].Not(), moveWolfBack[t].Not()])
        model.AddBoolOr([moveCabbageAccross[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveCabbageAccross[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveCabbageAccross[t].Not(), moveAccross[t].Not()])
        model.AddBoolOr([moveCabbageAccross[t].Not(), moveBack[t].Not()])
        model.AddBoolOr([moveWolfBack[t].Not(), moveSheepBack[t].Not()])
        model.AddBoolOr([moveWolfBack[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveWolfBack[t].Not(), moveAccross[t].Not()])
        model.AddBoolOr([moveWolfBack[t].Not(), moveBack[t].Not()])
        model.AddBoolOr([moveSheepBack[t].Not(), moveCabbageBack[t].Not()])
        model.AddBoolOr([moveSheepBack[t].Not(), moveAccross[t].Not()])
        model.AddBoolOr([moveSheepBack[t].Not(), moveBack[t].Not()])
        model.AddBoolOr([moveCabbageBack[t].Not(), moveAccross[t].Not()])
        model.AddBoolOr([moveCabbageBack[t].Not(), moveBack[t].Not()])
        model.AddBoolOr([moveAccross[t].Not(), moveBack[t].Not()])

    # Additional constraints (wolf eats sheep, sheep eats cabbage)
    for t in range(maxT):
        model.AddBoolOr([WolfOnThisSide[t].Not(), 
                         SheepOnThisSide[t].Not()]).OnlyEnforceIf(FerrymanOnThisSide[t].Not())
        model.AddBoolOr([WolfOnOppositeSide[t].Not(), 
                         SheepOnOppositeSide[t].Not()]).OnlyEnforceIf(FerrymanOnOppositeSide[t].Not())
        model.AddBoolOr([SheepOnThisSide[t].Not(), 
                         CabbageOnThisSide[t].Not()]).OnlyEnforceIf(FerrymanOnThisSide[t].Not())
        model.AddBoolOr([SheepOnOppositeSide[t].Not(), 
                         CabbageOnOppositeSide[t].Not()]).OnlyEnforceIf(FerrymanOnOppositeSide[t].Not())
    
    variables = []
    for t in range(maxT-1):
        variables.append(
                {
#                 "Ferryman on this side": FerrymanOnThisSide[t],
#                 "Ferryman on opposite side": FerrymanOnOppositeSide[t],
#                 "Wolf on this side": WolfOnThisSide[t],
#                 "Wolf on opposite side": WolfOnOppositeSide[t],
#                 "Sheep on this side": SheepOnThisSide[t],
#                 "Sheep on opposite side": SheepOnOppositeSide[t],
#                 "Cabbagge on this side": CabbageOnThisSide[t],
#                 "Cabbage on opposite side": CabbageOnOppositeSide[t],
                 "move accross":moveAccross[t],
                 "move back":moveBack[t],
                 "move wolf accross":moveWolfAccross[t],
                 "move wolf back":moveWolfBack[t],
                 "move sheep accross":moveSheepAccross[t],
                 "move sheep back": moveSheepBack[t],
                 "move cabbage accross": moveCabbageAccross[t],
                 "move cabbage back": moveCabbageBack[t]              
                 })
    
    solver = cp_model.CpSolver()    
    solver.SearchForAllSolutions(model, SolutionPrinter(variables))
    
    for t in range(maxT-1):
        if solver.Value(moveWolfAccross[t]): print(t, "move wolf accross")
        if solver.Value(moveWolfBack[t]): print(t, "move wolf back")
        if solver.Value(moveSheepAccross[t]): print(t, "move sheep accross")
        if solver.Value(moveSheepBack[t]): print(t, "move sheep back")
        if solver.Value(moveCabbageAccross[t]): print(t, "move cabbage accross")
        if solver.Value(moveCabbageBack[t]): print(t, "move cabbage back")
        if solver.Value(moveAccross[t]): print(t, "move accross")
        if solver.Value(moveBack[t]): print(t, "move back")
    
main()
