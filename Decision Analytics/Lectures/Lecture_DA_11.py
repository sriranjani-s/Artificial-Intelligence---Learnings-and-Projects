from ortools.sat.python import cp_model

num_staff = 10
num_shifts = 3
num_days = 7

staff_present_per_shift = 3
max_shifts_per_day = 1

#class SolutionPrinter(cp_model.CpSolverSolutionCallback):
#    def __init__(self, shifts):
#        cp_model.CpSolverSolutionCallback.__init__(self)
#        self.shifts_ = shifts
#        self.solutions_ = 0
#
#    def OnSolutionCallback(self):    
#        self.solutions_ = self.solutions_ + 1
#        print("Solution",self.solutions_)
#        for day in range(num_days):
#            print("Day ",day)
#            for shift in range(num_shifts):
#                print(" Shift",shift)
#                for staff in range(num_staff):
#                    if self.Value(self.shifts_[(staff,day,shift)]):
#                        print("  Present: ",staff)
#        print()

def main():
    model = cp_model.CpModel()

    shifts = {}
    for staff in range(num_staff):
        for day in range(num_days):
            for shift in range(num_shifts):
                shifts[(staff,day,shift)] = model.NewBoolVar("Shift"+str(staff)+"_"+str(day)+"_"+str(shift))

    for day in range(num_days):
        for shift in range(num_shifts):
            staff_present = []
            for staff in range(num_staff):
                staff_present.append(shifts[(staff,day,shift)])
            model.Add(sum(staff_present) == staff_present_per_shift)
    

    for day in range(num_days):
        for staff in range(num_staff):
            shifts_worked_per_day = []
            for shift in range(num_shifts):
                shifts_worked_per_day.append(shifts[(staff,day,shift)])
            model.Add(sum(shifts_worked_per_day) <= max_shifts_per_day)
                
    max_shifts_total = model.NewIntVar(0,num_days*num_shifts, "max_shifts_total")
    min_shifts_total = model.NewIntVar(0,num_days*num_shifts, "min_shifts_total")            
    for staff in range(num_staff):
        total_shifts_worked = []
        for day in range(num_days):
            for shift in range(num_shifts):
                total_shifts_worked.append(shifts[(staff,day,shift)])
        model.Add(sum(total_shifts_worked) <= max_shifts_total)
        model.Add(sum(total_shifts_worked) >= min_shifts_total)
    model.Minimize(max_shifts_total - min_shifts_total)
        
    solver = cp_model.CpSolver()    
    status = solver.Solve(model)

    if status==cp_model.OPTIMAL:        
        for day in range(num_days):
            print("Day ",day)
            for shift in range(num_shifts):
                print(" Shift",shift)
                for staff in range(num_staff):
                    if solver.Value(shifts[(staff,day,shift)]):
                        print("  Present: ",staff)
            print()
        print()
        
        print("Min shifts total:", solver.Value(min_shifts_total))
        print("Max shifts total:", solver.Value(max_shifts_total))
    else:
        print("No solution!")

    
main()
