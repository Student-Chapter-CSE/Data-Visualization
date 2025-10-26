import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_grade(average):
    """Calculate grade based on average marks"""
    if average >= 90:
        return 'A+'
    elif average >= 80:
        return 'A'
    elif average >= 70:
        return 'B'
    elif average >= 60:
        return 'C'
    elif average >= 50:
        return 'D'
    else:
        return 'F'

def main():
    """
    Main function to get student data from the user and print a summary.
    Returns the list of student data for visualization.
    """
    print("=== Student Marks and Grades Summary System ===\n")

    # Get number of students
    try:
        num_students = int(input("Enter number of students: "))
        if num_students <= 0:
            print("Please enter a positive number.")
            return [] # Return an empty list to signal no data
    except ValueError:
        print("Invalid input. Please enter a number.")
        return [] # Return an empty list

    students_data = []
    all_totals = []

    # Input data for each student
    for i in range(num_students):
        print(f"\n--- Student {i+1} ---")
        name = input("Enter student name: ")

        try:
            print("Enter marks for 3 subjects (out of 100):")
            subject1 = float(input("Subject 1: "))
            subject2 = float(input("Subject 2: "))
            subject3 = float(input("Subject 3: "))
            
            # Basic validation for marks
            if not (0 <= subject1 <= 100 and 0 <= subject2 <= 100 and 0 <= subject3 <= 100):
                print("Invalid mark. Marks should be between 0 and 100. Skipping this student.")
                continue

        except ValueError:
            print("Invalid mark. Please enter numbers only. Skipping this student.")
            continue # Skip to the next iteration of the loop

        # Calculate total and average
        total = subject1 + subject2 + subject3
        average = total / 3
        grade = calculate_grade(average)

        # Store student data
        student_info = {
            'name': name,
            'marks': [subject1, subject2, subject3],
            'Subject 1': subject1,  # Added key for easier plotting
            'Subject 2': subject2,  # Added key for easier plotting
            'Subject 3': subject3,  # Added key for easier plotting
            'total': total,
            'average': average,
            'grade': grade
        }

        students_data.append(student_info)
        all_totals.append(total)

    # Check if any data was actually added
    if not students_data:
        print("\nNo student data was successfully entered.")
        return []

    # Calculate class statistics
    actual_num_students = len(students_data)
    class_total = sum(all_totals)
    class_average = class_total / (actual_num_students * 3)

    # Find topper
    topper = max(students_data, key=lambda x: x['total'])

    # Display text summary
    print("\n" + "="*60)
    print("STUDENT MARKS AND GRADES SUMMARY")
    print("="*60)

    print(f"{'Name':<15} {'Sub1':<6} {'Sub2':<6} {'Sub3':<6} {'Total':<7} {'Avg':<6} {'Grade':<5}")
    print("-" * 60)

    for student in students_data:
        print(f"{student['name']:<15} "
              f"{student['marks'][0]:<6.1f} "
              f"{student['marks'][1]:<6.1f} "
              f"{student['marks'][2]:<6.1f} "
              f"{student['total']:<7.1f} "
              f"{student['average']:<6.1f} "
              f"{student['grade']:<5}")

    print("-" * 60)
    print(f"\nCLASS STATISTICS:")
    print(f"Total Students: {actual_num_students}")
    print(f"Class Average: {class_average:.2f}")
    print(f"Class Topper: {topper['name']} (Total: {topper['total']:.1f}, Average: {topper['average']:.1f})")

    # Grade distribution (text)
    grade_count = {}
    for student in students_data:
        grade = student['grade']
        grade_count[grade] = grade_count.get(grade, 0) + 1

    print(f"\nGRADE DISTRIBUTION:")
    for grade, count in sorted(grade_count.items()):
        print(f"Grade {grade}: {count} student(s)")
        
    # Return the collected data for the next function
    return students_data

def create_visualizations(students_data):
    """
    Generates and displays charts based on the student data.
    """
    if not students_data:
        print("\nCannot create visualizations: No student data available.")
        return

    # Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(students_data)
    
    # Set a nice style for the plots
    sns.set_theme(style="whitegrid")
    
    # --- Chart 1: Grade Distribution (Bar Chart) ---
    plt.figure(figsize=(10, 6))
    grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']
    sns.countplot(data=df, x='grade', order=grade_order, palette="viridis")
    plt.title('Class Grade Distribution', fontsize=16)
    plt.xlabel('Grade', fontsize=12)
    plt.ylabel('Number of Students', fontsize=12)
    
    # --- Chart 2: Distribution of Total Scores (Histogram) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='total', kde=True, bins=10, color="blue")
    plt.title('Distribution of Total Scores', fontsize=16)
    plt.xlabel('Total Score (out of 300)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # --- Chart 3: Score Distribution by Subject (Box Plot) ---
    
    # "Melt" the DataFrame to prepare it for Seaborn
    df_melted = df.melt(id_vars=['name', 'grade'], 
                          value_vars=['Subject 1', 'Subject 2', 'Subject 3'],
                          var_name='Subject', 
                          value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x='Subject', y='Score', palette="pastel")
    
    plt.title('Score Distribution by Subject', fontsize=16)
    plt.xlabel('Subject', fontsize=12)
    plt.ylabel('Score (out of 100)', fontsize=12)
    
    # Finally, display all the plots
    print("\nGenerating visualizations... Close the plot windows to exit.")
    plt.tight_layout() # Adjusts plots to prevent overlap
    plt.show()


# ==================================================================
# This is the main entry point that runs the program
# ==================================================================
if __name__ == "__main__":
    
    # 1. Run the main data entry and summary function
    # This will ask for user input and print the text summary
    student_data_list = main()
    
    # 2. Pass the collected data to the new visualization function
    # This will only run if data was successfully entered (list is not empty)
    if student_data_list: 
        create_visualizations(student_data_list)
    else:
        print("\nExiting program.")