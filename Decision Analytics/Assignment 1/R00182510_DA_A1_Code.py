from ortools.sat.python import cp_model
import pandas as pd
import numpy as np

# Task1 global variables
# Task1 - Objects
students = ["Student #1", "Student #2", "Student #3", "Student #4"]

# Task1 - Predicates an Attributes
names = ["Carol", "Elisa", "Oliver", "Lucas"]
universities = ["London", "Cambridge", "Oxford", "Edinburgh"]
genders = ["boy", "girl"]
nationalities = ["Australia", "USA", "South Africa", "Canada"]
majors = ["History", "Medicine", "Law", "Architecture"]


# Task1 Solution printer to print the optimized details for each student
class Task1_SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, student_nationality, student_university, student_name, student_major, student_gender):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.nationalities = student_nationality
        self.universities = student_university
        self.names = student_name
        self.majors = student_major
        self.genders = student_gender
        self.solutions_ = 0

    def OnSolutionCallback(self):
        self.solutions_ = self.solutions_ + 1
        print("Solution", self.solutions_)

        for student in students:
            for name in names:
                if self.Value(self.names[student][name]):
                    print(student + " is " + name)
            for nationality in nationalities:
                if self.Value(self.nationalities[student][nationality]):
                    print(student + " is from " + nationality)

            for university in universities:
                if self.Value(self.universities[student][university]):
                    print(student + " goes to " + university)

            for major in majors:
                if self.Value(self.majors[student][major]):
                    print(student + " studies " + major)

            for gender in genders:
                if self.Value(self.genders[student][gender]):
                    print(student + " is " + gender)
        print()


# Task 1 - Creating decision variables and assigning constraints
def Task1():
    model = cp_model.CpModel()

    # Subtask A - creating decision variables based on the predicates and attributes
    student_name = {}
    for student in students:
        variables = {}
        for name in names:
            variables[name] = model.NewBoolVar(student + name)
        student_name[student] = variables

    student_university = {}
    for student in students:
        variables = {}
        for university in universities:
            variables[university] = model.NewBoolVar(student + university)
        student_university[student] = variables

    student_gender = {}
    for student in students:
        variables = {}
        for gender in genders:
            variables[gender] = model.NewBoolVar(student + gender)
        student_gender[student] = variables

    student_nationality = {}
    for student in students:
        variables = {}
        for nationality in nationalities:
            variables[nationality] = model.NewBoolVar(student + nationality)
        student_nationality[student] = variables

    student_major = {}
    for student in students:
        variables = {}
        for major in majors:
            variables[major] = model.NewBoolVar(student + major)
        student_major[student] = variables
    # print(student_name, student_university, student_gender, student_nationality, student_major)

    # Subtask C - Implicit Constraints

    # 1) Every student has atleast has one name
    #    Every student has atleast one university
    #    Every student has atleast one nationality
    #    Every student has atleast one major
    #    Every student has atleast one gender

    for student in students:
        variables = []
        for name in names:
            variables.append(student_name[student][name])
        model.AddBoolOr(variables)

        variables = []
        for university in universities:
            variables.append(student_university[student][university])
        model.AddBoolOr(variables)

        variables = []
        for nationality in nationalities:
            variables.append(student_nationality[student][nationality])
        model.AddBoolOr(variables)

        variables = []
        for major in majors:
            variables.append(student_major[student][major])
        model.AddBoolOr(variables)

        variables = []
        for gender in genders:
            variables.append(student_gender[student][gender])
        model.AddBoolOr(variables)

    # 2) Every student has no more than one name
    #    Every student has no more than one university
    #    Every student has no more than one nationality
    #    Every student has no more than one major
    #    Every student has no more than one gender

    for student in students:
        for i in range(4):
            for j in range(i + 1, 4):
                model.AddBoolOr([student_name[student][names[i]].Not(),
                                 student_name[student][names[j]].Not()])

                model.AddBoolOr([student_university[student][universities[i]].Not(),
                                 student_university[student][universities[j]].Not()])

                model.AddBoolOr([student_nationality[student][nationalities[i]].Not(),
                                 student_nationality[student][nationalities[j]].Not()])

                model.AddBoolOr([student_major[student][majors[i]].Not(),
                                 student_major[student][majors[j]].Not()])
        for i in range(2):
            for j in range(i + 1, 2):
                model.AddBoolOr([student_gender[student][genders[i]].Not(),
                                 student_gender[student][genders[j]].Not()])

    # 3) Every student has a different name
    #    Every student has a different university
    #    Every student has a different nationality
    #    Every student has a different major

    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(4):
                model.AddBoolOr([student_name[students[i]][names[k]].Not(),
                                 student_name[students[j]][names[k]].Not()])

                model.AddBoolOr([student_university[students[i]][universities[k]].Not(),
                                 student_university[students[j]][universities[k]].Not()])

                model.AddBoolOr([student_nationality[students[i]][nationalities[k]].Not(),
                                 student_nationality[students[j]][nationalities[k]].Not()])

                model.AddBoolOr([student_major[students[i]][majors[k]].Not(),
                                 student_major[students[j]][majors[k]].Not()])

    # 4) There are two boys (Oliver, Lucas) and two girls (Elisa, Carol)
    # Gender(student, girl) -> Name('Elisa') or Name('Carol')
    # Gender(student, boy) -> Name('Oliver') or Name('Lucas')

    for student in students:
        model.AddBoolOr([student_name[student]["Elisa"], student_name[student]['Carol']]).OnlyEnforceIf(
            student_gender[student]["girl"])
        model.AddBoolOr([student_name[student]["Oliver"], student_name[student]['Lucas']]).OnlyEnforceIf(
            student_gender[student]["boy"])


    # Subtask B - Explicit Sentence Constraints

    # 1) One of them is going to London.
    # University(s1, London) -> !University(s2,London) & !University(s3, London) & !University(s4, London)
    # University(s2, London) -> !University(s1,London) & !University(s3, London) & !University(s4, London)
    # University(s3, London) -> !University(s1,London) & !University(s2, London) & !University(s4, London)
    # University(s34, London) -> !University(s1,London) & !University(s2, London) & !University(s3, London)

    for i, student_i in enumerate(students):
        other = [value for j, value in enumerate(students) if j != i]
        for j in range(len(other)):
            model.AddBoolAnd([student_university[other[j]]["London"].Not()]).OnlyEnforceIf(
                student_university[student_i]["London"])

    # 2) Exactly one boy and one girl chose a university in a city with the same initial of their names
    # University('Carol', 'Cambridge') -> !University('Elisa', 'Edinburgh')
    # University('Elisa', 'Edinburgh') -> !University('Carol', 'Cambridge')
    # University('Oliver', 'Oxford') -> !University('Lucas', 'London')
    # University('Lucas', 'London')  -> !University('Oliver', 'Oxford')

    # Gender(student, 'girl'), Name(student, 'Carol'), University(student, Cambridge) -> !University(other, 'Edinburgh'), Gender(other, 'girl'), Name(other, 'Elisa')
    # Gender(student, 'girl'), Name(student, 'Elisa'), University(student, Edinburgh) -> !University(other, 'Cambridge'), Gender(other, 'girl'), Name(other, 'Carol')
    # Gender(student, 'boy'), Name(student, 'Lucas'), University(student, London) -> !University(other, 'Oxford'), Gender(other, 'boy'), Name(other, 'Oliver')
    # Gender(student, 'boy'), Name(student, 'Oliver'), University(student, Oxford) -> !University(other, 'London'), Gender(other, 'boy'), Name(other, 'Lucas')

    for i, student_i in enumerate(students):

        other = [value for j, value in enumerate(students) if j != i]
        for j in range(len(other)):
            model.AddBoolAnd([student_name[other[j]]["Elisa"],
                              student_university[other[j]]["Edinburgh"].Not()]).OnlyEnforceIf([student_name[student_i]["Carol"],
                                                                                      student_university[student_i]["Cambridge"],
                                                                                      student_gender[student_i]["girl"],
                                                                                      student_gender[other[j]]["girl"]])

            model.AddBoolAnd([student_name[other[j]]["Elisa"],
                              student_university[other[j]]["Edinburgh"]]).OnlyEnforceIf([student_name[student_i]["Carol"],
                                                                                      student_university[student_i]["Cambridge"].Not(),
                                                                                      student_gender[student_i]["girl"],
                                                                                      student_gender[other[j]]["girl"]])

            model.AddBoolAnd([student_name[other[j]]["Carol"],
                              student_university[other[j]]["Cambridge"].Not()]).OnlyEnforceIf([student_name[student_i]["Elisa"],
                                                                                      student_university[student_i]["Edinburgh"],
                                                                                      student_gender[student_i]["girl"],
                                                                                      student_gender[other[j]]["girl"]])

            model.AddBoolAnd([student_name[other[j]]["Carol"],
                              student_university[other[j]]["Cambridge"]]).OnlyEnforceIf([student_name[student_i]["Elisa"],
                                                                                      student_university[student_i]["Edinburgh"].Not(),
                                                                                      student_gender[student_i]["girl"],
                                                                                      student_gender[other[j]]["girl"]])

            model.AddBoolAnd([student_name[other[j]]["Oliver"],
                              student_university[other[j]]["Oxford"].Not()]).OnlyEnforceIf([student_name[student_i]["Lucas"],
                                                                                      student_university[student_i]["London"],
                                                                                      student_gender[student_i]["boy"],
                                                                                      student_gender[other[j]]["boy"]])

            model.AddBoolAnd([student_name[other[j]]["Oliver"],
                              student_university[other[j]]["Oxford"]]).OnlyEnforceIf([student_name[student_i]["Lucas"],
                                                                                      student_university[student_i]["London"].Not(),
                                                                                      student_gender[student_i]["boy"],
                                                                                      student_gender[other[j]]["boy"]])

            model.AddBoolAnd([student_name[other[j]]["Lucas"],
                              student_university[other[j]]["London"]]).OnlyEnforceIf([student_name[student_i]["Oliver"],
                                                                                      student_university[student_i]["Oxford"].Not(),
                                                                                      student_gender[student_i]["boy"],
                                                                                      student_gender[other[j]]["boy"]])

            model.AddBoolAnd([student_name[other[j]]["Lucas"],
                              student_university[other[j]]["London"].Not()]).OnlyEnforceIf([student_name[student_i]["Oliver"],
                                                                                      student_university[student_i]["Oxford"],
                                                                                      student_gender[student_i]["boy"],
                                                                                      student_gender[other[j]]["boy"]])


    # 3) A boy is from Australia, the other studies History
    # Gender(student, 'girl') -> !Nationality(student, 'Australia')
    # Gender(student, 'girl') -> !Major(student, 'History')
    # Gender(student, 'boy') -> !( Major(student, 'History') AND  Nationality(student, 'Australia') )
    #                        -> !Major(student, 'History') OR !Nationality(student, 'Australia') )

    for student in students:
        model.AddBoolOr([student_major[student]['History'].Not(),
                         student_nationality[student]['Australia'].Not()]).OnlyEnforceIf(student_gender[student]['boy'])

        model.AddBoolOr([student_nationality[student]['Australia'].Not()]).OnlyEnforceIf(
            student_gender[student]['girl'])
        model.AddBoolOr([student_major[student]['History'].Not()]).OnlyEnforceIf(student_gender[student]['girl'])


    # 4) A girl goes to Cambridge, the other studies Medicine
    # Gender(student, 'boy') -> !University(student, 'Cambridge')
    # Gender(student, 'boy') -> !Major(student, 'Medicine')
    # Gender(student, 'girl') -> !( (University(other, 'Cambridge') AND Major(other, 'Medicine') )
    #                         -> !University(other, 'Cambridge') OR !Major(other, 'Medicine')

    for student in students:
        model.AddBoolAnd([student_university[student]['Cambridge'].Not(),
                          student_major[student]['Medicine'].Not()]).OnlyEnforceIf(student_gender[student]['boy'])

        model.AddBoolOr([student_major[student]['Medicine'].Not(),
                         student_university[student]['Cambridge'].Not()]).OnlyEnforceIf(student_gender[student]['girl'])


    # 5) Oliver studies Law or is from USA; He is not from South Africa
    # Name(student, 'Oliver') -> !Nationality(student, 'South Africa')
    # Name(student, 'Oliver') -> Major(student, 'Law') OR Nationality(student, 'USA')

    for student in students:
        model.AddBoolAnd([student_nationality[student]['South Africa'].Not()]).OnlyEnforceIf(
            student_name[student]['Oliver'])
        model.AddBoolOr([student_major[student]['Law'],
                         student_nationality[student]['USA']]).OnlyEnforceIf(student_name[student]['Oliver'])


    # 6) The student from Canada is a historian or will go to Oxford
    # Nationality(student, 'Canada') -> Major(student, 'Historian') OR University(student, 'Oxford')

    for student in students:
        model.AddBoolOr([student_major[student]['History'],
                         student_university[student]['Oxford']]).OnlyEnforceIf(student_nationality[student]['Canada'])


    # 7) The student from South Africa is going to Edinburgh or will study Law
    # Nationality(student, 'South Africa') -> Major(student, 'Law') OR University(student, 'Edinburgh')

    for student in students:
        model.AddBoolOr([student_major[student]['Law'],
                         student_university[student]['Edinburgh']]).OnlyEnforceIf(
            student_nationality[student]['South Africa'])


    #Additional constraint
    # Recieved multiple solutions with the same assignment of values for all variables but with different ordering
    # of students. To overcome this and receive only one solution, the student number is assigned the names in one order

    model.AddBoolAnd([student_name['Student #1']['Lucas']])
    model.AddBoolAnd([student_name['Student #2']['Carol']])
    model.AddBoolAnd([student_name['Student #3']['Oliver']])
    model.AddBoolAnd([student_name['Student #4']['Elisa']])

    # Subtask D - Solve the model with the above constraints
    solver = cp_model.CpSolver()
    solver.SearchForAllSolutions(model,
                                 Task1_SolutionPrinter(student_nationality, student_university, student_name, student_major,
                                                 student_gender))

    for student in students:
        if solver.Value(student_major[student]['Architecture']):
            for nationality in nationalities:
                if solver.Value(student_nationality[student][nationality]):
                    print("The Nationality of the Architecture student is: ", nationality)


# Task2 Solution printer to print the projects taken and assignment details for each project.
# The maximum gained profit is also printed
class Task2_SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, projects_taken, contracts, projects, jobs_list, months_list, contractors_list, Quotes, Proj_values):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.projects_taken = projects_taken
        self.contracts = contracts
        self.projects = projects
        self.jobs = jobs_list
        self.months = months_list
        self.contractors = contractors_list
        self.Quotes = Quotes
        self.Proj_values = Proj_values
        self.solutions_ = 0

    def OnSolutionCallback(self):
        self.solutions_ = self.solutions_ + 1
        print("Solution", self.solutions_ )

        projects = []
        total_cost = 0

        # Print the project taken to be delivered
        for i, project in enumerate(self.projects):
            if self.Value(self.projects_taken[project]):
                projects.append(project)

        print('Projects Taken: ', projects)

        # Print the assignment details for each project
        print()
        print('Project Assignments:')
        for project in projects:
            for month in self.months:
                for job in self.jobs:
                    for contractor in self.contractors:
                        if ((project, month, job, contractor)) in self.contracts.keys():
                            if self.Value(self.contracts[(project, month, job, contractor)]):
                                contractor_cost = int(self.Quotes.loc[contractor, job])
                                print('Project, Month, Job, Contractor, cost: ', (project, month, job, contractor, contractor_cost))
                                total_cost += contractor_cost

        # Print the overall project value, contractor costs and the profit gained
        Proj_value = sum(int(self.Proj_values.loc[project]) for project in projects)
        profit = Proj_value - total_cost
        print()
        print('Overall project value: ', Proj_value)
        print('Overall contractor cost: ', total_cost)
        print('Gained Profit: ', profit)
        print()
        print()


# Task 2 - Creating decision variables and assigning constraints
def Task2():

    # Subtask A - Data imported from provided excel sheet
    data = pd.read_excel("Assignment_DA_1_data.xlsx", sheet_name=None)
    Projects = data["Projects"]
    Quotes = data["Quotes"]
    Dependencies = data["Dependencies"]
    Value = data["Value"]

    model = cp_model.CpModel()

    projects = Projects['Unnamed: 0']
    contractors_list = Quotes['Unnamed: 0']
    jobs_list = list(Quotes.columns[1:])
    months_list = list(Projects.columns[1:])

    # Extract all the valid data from the input excel sheet
    # extract only the valid months for each project
    months = {}
    for i in range(len(projects)):
        indices = np.where(Projects.iloc[i, 1:].notnull())
        months[projects[i]] = list(Projects.columns[1:][indices])

    # extract only the valid jobs for each project and month
    jobs = {}
    i = 0
    for project in projects:
        indices = list((np.where(Projects.iloc[i, :].notnull())[0]))
        indices.pop(0)
        j = 0
        for month in months[project]:
            variable = []
            variable.append(Projects.iloc[i, indices[j]])
            jobs[(project, month)] = variable
            j += 1
        i += 1

    # extract only the valid contractors for each project, month and job
    contractors = {}
    for i, project in enumerate(projects):
        for month in months[project]:
            job = jobs[(project, month)][0]
            indices = Quotes[job].dropna().index
            contractors[(project, month, job)] = contractors_list[indices].tolist()


    # Subtask B - Create a boolean decision variable for choosing the projects to deliver
    projects_taken = {}
    for project in projects:
        projects_taken[project] = model.NewBoolVar(project)


    # Subtask B - Create a Boolean decision variable for the allowed combinations of (project, month, job and contractor)
    # as extracted above
    contract = {}
    for project in projects:
        for month in months[project]:
            for job in jobs[(project, month)]:
                for contractor in contractors[(project, month, job)]:
                    contract[(project, month, job, contractor)] = model.NewBoolVar("Contract" + str(project) +
                                                                                   '_' + str(month) +
                                                                                   '_' + str(job) +
                                                                                   '_' + str(contractor))
    # print(contract)

    # Subtask B - Constraint to ensure that if a project is taken, it runs for only the required months
    for project in projects:
        for month in months[project]:
            variables = []
            for job in jobs[(project, month)]:
                for contractor in contractors[(project, month, job)]:
                    variables.append(contract[(project, month, job, contractor)])
            model.AddBoolOr(variables).OnlyEnforceIf(projects_taken[project])

    # Subtask C - a contractor cannot work on two projects simultaneously
    for month in months_list:
        for contractor in contractors_list:
            contractor_projects_taken = []
            for job in jobs_list:
                for project in projects:
                    if ((project, month, job, contractor)) in contract.keys():
                        contractor_projects_taken.append(contract[(project, month, job, contractor)])
                    else:
                        continue
                model.Add(sum(contractor_projects_taken) <= 1)

    # Subtask D - if a project is accepted to be delivered then exactly one contractor per job of the project needs to work on it
    #      AND
    # Subtask E - if a project is not taken on then no one should be contracted to work on it
    max_contractor_jobs_taken = 1
    no_contractors_taken = 0

    for project in projects:
        for month in months_list:
            for job in jobs_list:
                contractor_jobs_taken = []
                for contractor in contractors_list:
                    if ((project, month, job, contractor)) in contract.keys():
                        contractor_jobs_taken.append(contract[(project, month, job, contractor)])
                    else:
                        continue
                # Constraint D
                model.Add(sum(contractor_jobs_taken) <= max_contractor_jobs_taken).OnlyEnforceIf(projects_taken[project])
                # Constraint E
                model.Add(sum(contractor_jobs_taken) == no_contractors_taken).OnlyEnforceIf(projects_taken[project].Not())

    # Subtask F - the project dependency constraints
    dependency = {}

    for i in range(len(Dependencies)):
        indices = np.where(Dependencies.iloc[i, 1:].notnull())
        dep_proj = list(Dependencies.columns[1:][indices])
        if len(dep_proj) > 0:
            dependency[projects[i]] = dep_proj

    for key, values in dependency.items():
        for item in values:
            model.AddBoolAnd([projects_taken[key].Not()]).OnlyEnforceIf(projects_taken[item].Not())

    # Subtask G -  the profit margin between the value of all delivered projects and the cost of all required
    # subcontractors, is at least €2500
    total_contractor_cost = 0
    minimum_profit = 2500

    Quotes.set_index('Unnamed: 0', inplace=True)
    Value.set_index('Unnamed: 0', inplace=True)

    for project in projects:
        for month in months_list:
            for job in jobs_list:
                for contractor in contractors_list:
                    if ((project, month, job, contractor)) in contract.keys():
                        total_contractor_cost += (
                                    contract[(project, month, job, contractor)] * int(Quotes.loc[contractor, job]))

    model.Add((sum(projects_taken[project] * int(Value.loc[project])
                   for project in projects) - total_contractor_cost) >= minimum_profit)


    # Subtask F - Solve the model with the above constraints
    solver = cp_model.CpSolver()
    status = solver.SearchForAllSolutions(model,
                                          Task2_SolutionPrinter(projects_taken, contract, projects, jobs_list, months_list,
                                                          contractors_list, Quotes, Value))
    print(solver.StatusName(status))



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
