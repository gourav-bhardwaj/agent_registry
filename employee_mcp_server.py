from mcp.server.fastmcp import FastMCP
from typing import Literal
from pydantic import BaseModel

mcp = FastMCP("employee_mcp_server")

class Employee(BaseModel):
    name: str
    personal_email: str
    official_email: str
    address: str
    department: Literal["Engineering", "HR", "IT Support"]

@mcp.tool(name="onboard_employee", description="Insert the employee into database as a record")
def onboard_employee(name: str, personal_email: str, official_email: str, address: str, department: Literal["Engineering", "HR", "IT Support"]):
    try:
        if not name or not personal_email:
            raise Exception("name and personal email are required fields")
        print(f"""
        Employee Name: {name}
        Personal Email: {personal_email}
        Official Email: {official_email}
        Address: {address}
        Department: {department}
        """)
        return {"status": 201, "message": "Employee has been onboarded successfully"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}
    
@mcp.tool(name="deboard_employee", description="Delete the particular employee from database")
def deboard_employee(email: str):
    try:
        if not email:
            raise Exception("DEBOARD_EMPLOYEE: official email is required field")
        print(f"""
        Personal Email: {email}
        """)
        return {"status": 201, "message": "Employee has been deboarded successfully"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}

@mcp.tool(name="get_employee", description="Get the particular employee")
def get_employee(email: str):
    try:
        if not email:
            raise Exception("GET_EMPLOYEE: official email is required field")
        print(f"""
        Personal Email: {email}
        """)
        return {"status": 201, "message": "Employee has been fetched successfully", "response": Employee(name=email.split("@")[0], personal_email=email, official_email=email, address="Abc", department="Engineering")}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}

@mcp.tool(name="validate_employee_details", description="Validate employee details")
def validate_employee_details(email: str):
    try:
        if not email:
            raise Exception("VALIDATE_EMPLOYEE_DETAILS: official email is required field")
        print(f"""
        Personal Email: {email}
        """)
        return {"status": 201, "message": "Employee has been validated successfully"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")