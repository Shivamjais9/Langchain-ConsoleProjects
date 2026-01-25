from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class Student(BaseModel):

    Name: str = 'Shivam'
    Age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=6)

new_student = {'age':'21', 'email':'abc@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)
print(student_dict)