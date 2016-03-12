# Users dictionary and modularisation

users={"Arslan":"1234","Ufi":"5678"}

students={"Gazal":[1,2,3,4],"Khurram":[1,2,3,4]}

def auth():
    user=raw_input("Username>")
    pswd=raw_input("Password>")
    if user in users:
        if users[user]==pswd:
            return 1
        else:
            print "Access Denied"
            
            return 0
    else:
        print "Access Denied"
             
            
            
            

def addGrade():
    student=raw_input("Enter students name>")
    add_grade=raw_input("Enter Grade>")
    if student in students:
        students[student].append(int(add_grade))
    else:
        add_choice=raw_input("Student does not exist, do you want add the student?")
        if add_choice =="y":
            addStudent(student,int(add_grade))
        else:    
            print "exiting!"
            
        
        

def removeStudent():
    student=raw_input("Enter students name>")
    if student in students:
        print "Are you sure you want to delete",student,"?"
        l_choice=raw_input()
        if l_choice=="y":
            del students[student]
        else:
            print"Student deleted"
    else:
        print "Student does not exist"

def viewAvgGrade():
    student=raw_input("Enter students name>")
    if student in students:
        avg=sum(students[student])
        print "The total of %s is %d." % (student,avg)
    else:
        print "The student does not exist"        
        

def addStudent(student,grade):
    students[student]=grade



def choices():
    print """
    1 : Add Grade
    2 : View Average grade
    3 : Remove student
    4 : Exit        
    """
    choice = raw_input("Enter your choice buddy >")
    return choice
    
def processing(c):
    print"Your entered choice is",c    
    if c =="1":
        addGrade()
    elif c=="2":
        viewAvgGrade()
    elif c=="3":
        removeStudent()
    elif c=="4":
        print "Exiting"
    else:
        print ("Invalid choice entered.")    
    
#Program starts here    
    
    
c=0
while auth()==1:
    c=choices()
    processing(c)


