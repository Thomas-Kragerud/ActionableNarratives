import base64
import os
from enum import Enum
from io import BytesIO
from typing import Iterable, List, Literal, Optional, Union


import fitz
from dotenv import load_dotenv

# Instructor is powered by Pydantic, which is powered by type hints.
# Schema validation, prompting is controlled by type annotations
import instructor
import matplotlib.pyplot as plt
import pandas as pd
#from IPython.display import display
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field


# Function to encode image as base64
def encode_image(image_path: str):
    # check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def display_images(image_data: dict):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, (key, value) in enumerate(image_data.items()):
        img = Image.open(BytesIO(base64.b64decode(value)))
        ax = axs[i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(key)
    plt.tight_layout()
    plt.show()


class Order(BaseModel):
    """ Represents an order with details such as order ID, customer name, product name, price, status
    and delivery data
    The ... is ellipsis and used to indicate that a field is required. This means that the field must be provided when
    creating an instance of the model, and if not a validation error will occur.
    """
    order_id: str = Field(..., description="The unique identifier of the order")
    product_name: str = Field(..., description="The name of the product")
    price: float = Field(..., description="The price of the product")
    status: str = Field(..., description="The status of the order")
    delivery_date: str = Field(..., description="The delivery date of the order")


# Placeholder functions for order processing

def get_order_details(order_id):
    # Placeholder function to retrieve order details based on the order ID
    return Order(
        order_id=order_id,
        product_name="Product X",
        price=100.0,
        status="Delivered",
        delivery_date="2024-04-10",
    )

def escalate_to_agent(order: Order, message: str):
    # Placeholder function to escalate the order to a human agent
    return f"Order {order.order_id} has been escalated to an agent with message: `{message}`"

def refund_order(order: Order):
    # Placeholder function to process a refund for the order
    return f"Order {order.order_id} has been refunded successfully."

def replace_order(order: Order):
    # Placeholder function to replace the order with a new one
    return f"Order {order.order_id} has been replaced with a new order."


class FunctionCallBase(BaseModel):
    """
    Literals: way to indicate that a variable or parameter can have only specific literal values
    When you use literal in type hints, you are explicitly stating the exact values that a variable can be assigned

    In Pydantic, fields are defined at the class leve, specifying the type and validation requirements for each field.
    These definitions aren't variable assignments but rather declarations of the structure of the data and its constraints.

    When you create an instance of this class FunctionCallBase, Pydantic uses these class-level declarations to
    dynamically create instance variables. You access these fields using self within the methods to refer to the
    instance-level data specific to that object

    Pydantics BaseModel automatically generates an __init__ method based on the fields you declare in the class body.
    """
    rationale: Optional[str] = Field(..., description="The reason for the action.")
    image_description: Optional[str] = Field(
        ..., description="The detailed description of the package image."
    )
    action: Literal["escalate_to_agent", "replace_order", "refund_order"]
    message: Optional[str] = Field(
        ...,
        description="The message to be escalated to the agent if action is escalate_to_agent",
    )

    # Placeholder functions to process the action based on the order ID
    def __call__(self, order_id):
        order: Order = get_order_details(order_id=order_id)
        if self.action == "escalate_to_agent":
            return escalate_to_agent(order, self.message)
        if self.action == "replace_order":
            return replace_order(order)
        if self.action == "refund_order":
            return refund_order(order)


class EscalateToAgent(FunctionCallBase):
    """Escalate to an agent for further assistance."""
    pass

class OrderActionBase(FunctionCallBase):
    pass

class ReplaceOrder(OrderActionBase):
    """Tool call to replace an order."""
    pass

class RefundOrder(OrderActionBase):
    """Tool call to refund an order."""
    pass

def delivery_exception_support_handler(test_image: str):
    """
    response model: the expected structure of the response from the model. In this case, its specified as Iterable
    which means that the response should be a list containing instances of RefundOrder, ReplaceOrder or EscalateToAgent

    tool_choice: Is set to "auto" meaning the model should automatically select the appropriate tool based on the context

    messages: is updated with two elements.
    The first element is a dictionary representing the instruction prompt. It has the role of user and contend
    filed containing the instruction prompt
    The second element is another dictionary representing the image data. It has role of user and content filed
    containing a list with a single dictionary. The dictionary has type image_url and image url filed containing the
    base 64 encoded image data

    """
    payload = {
        "model": MODEL,
        # When receiving a response from gpt4 Instructor validates the response against the specified response model
        "response_model": Iterable[Union[RefundOrder, ReplaceOrder, EscalateToAgent]],
        "tool_choice": "auto",  # automatically select the tool based on the context
        "temperature": 0.0,  # for less diversity in responses
        "seed": 123,  # Set a seed for reproducibility
    }

    payload["messages"] = [
        {
            "role": "system",
            "content": INSTRUCTION_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data[test_image]}"
                    }
                },
            ],
        }
    ]
    function_calls = instructor.from_openai(
        OpenAI(api_key=openai_api_key), mode=instructor.Mode.PARALLEL_TOOLS
    ).chat.completions.create(**payload)
    for tool in function_calls:
        print(f"- Tool call: {tool.action} for provided img: {test_image}")
        print(f"- Parameters: {tool}")
        print(f">> Action result: {tool(ORDER_ID)}")
        return tool


# Code for example 2

# Function to convert a single page PDF page to a JPEG image
def convert_pdf_page_to_jpg(pdf_path: str, output_path: str, page_number=0):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap()
    pix.save(output_path)


def display_img_local(image_path: str):
    img = Image.open(image_path)
    img.show()


class RoleEnum(str, Enum):
    """ Defines possible roles within an organization """
    CEO = "CEO"
    CTO = "CTO"
    CFO = "CFO"
    COO = "COO"
    EMPLOYEE = "Employee"
    MANAGER = "Manager"
    INTERN = "Intern"
    OTHER = "Other"

class Employee(BaseModel):
    """Represents an employee, including their name, role, and optional manager information."""
    employee_name: str = Field(..., description="The name of the employee")
    role: RoleEnum = Field(..., description="The role of the employee")
    manager_name: Optional[str] = Field(None, description="The manager's name, if applicable")
    manager_role: Optional[RoleEnum] = Field(None, description="The manager's role, if applicable")

class EmployeeList(BaseModel):
    """A list of employees within the organizational structure"""
    employees: List[Employee] = Field(..., description="A list of employees")

def parse_orgchart(base_img: str) -> EmployeeList:
    response = instructor.from_openai(OpenAI(api_key=openai_api_key)).chat.completions.create(
        model="gpt-4-turbo",
        response_model=EmployeeList,
        messages=[
            {
                "role": "system",
                "content": 'Analyze the given organizational chart and very carefully extract the information.',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    }
                ]
            }
        ]
    )
    return response


if __name__ == "__main__":
    # Loade .env variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    MODEL = "gpt-4-turbo-2024-04-09"

    # Example 1
    # =========================================000
    # Sample images for testing
    image_dir = "example_data/images"

    # encode all images within the dir
    image_fils = os.listdir(image_dir)
    image_data = {}
    for image_file in image_fils:
        image_path = os.path.join(image_dir, image_file)
        # encode the image with key as the image file name
        image_data[image_file.split(".")[0]] = encode_image(image_path)
        print(f"Encoded image: {image_file}")

    display_images(image_data)

    # ----------------------------------------------
    # extract the tool call from the response

    ORDER_ID = "12345"  # Placeholder order ID for testing
    INSTRUCTION_PROMPT = "You are a customer service assistant for a delivery service, equipped to analyze images of packages. If a package appears damaged in the image, automatically process a refund according to policy. If the package looks wet, initiate a replacement. If the package appears normal and not damaged, escalate to agent. For any other issues or unclear images, escalate to agent. You must always use tools!"

    # ----------------------------------------------
    print("Processing delivery exception support for different package images...")

    print("\n===================== Simulating user message 1 =====================")
    assert delivery_exception_support_handler("damaged_package").action == "refund_order"

    print("\n===================== Simulating user message 2 =====================")
    assert delivery_exception_support_handler("normal_package").action == "escalate_to_agent"

    print("\n===================== Simulating user message 3 =====================")
    assert delivery_exception_support_handler("wet_package").action == "replace_order"


    # Example 2 ==========================================

    pdf_path = "example_data/data/org-chart-sample.pdf"
    output_path = "org-chart-sample.jpg"

    convert_pdf_page_to_jpg(pdf_path, output_path)
    display_img_local(output_path)

    base64_img = encode_image(output_path)

    # call the functions to analyze the organizational chart and parse the respons
    result = parse_orgchart(base64_img)

    # tabulate the extracted data
    # tabulate the extracted data
    df = pd.DataFrame([{
        'employee_name': employee.employee_name,
        'role': employee.role.value,
        'manager_name': employee.manager_name,
        'manager_role': employee.manager_role.value if employee.manager_role else None
    } for employee in result.employees])
    print(df.to_string())
    

