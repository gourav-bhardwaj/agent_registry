from mcp.server.fastmcp import FastMCP
from typing import Literal
from pydantic import BaseModel
import logging

# 1. Base Configuration setup
logging.basicConfig(
    level=logging.INFO,                           # Set minimum severity level to log
    format='%(asctime)s - %(levelname)s - %(message)s', # Define log layout structure
    datefmt='%Y-%m-%d %H:%M:%S'                    # Style the timestamp output
)

# 2. Instantiate a module-level logger 
logger = logging.getLogger(__name__)


mcp = FastMCP("notification_mcp_server")

@mcp.tool(name="send_notification", description="Send notification to the particular recipient through sender")
def send_notification(sender: str, recipient: str, subject: str, message: str):
    try:
        if not sender or not message or not recipient:
            raise Exception("ONBOARD_EMPLOYEE: sender, recipient and message are required fields")

        if not sender.endswith("myorgemail.com"):
            raise Exception("ONBOARD_EMPLOYEE: sender is invalid!!")

        if not recipient.endswith("myorgemail.com"):
            raise Exception("ONBOARD_EMPLOYEE: recipient is invalid!!")
        logger.info(f"""
Action: Send Notification
Sender: {sender}
Recipient: {recipient}
Subject: {subject}
Message: {message}
""")
        return {"status": 200, "message": f"Notification has been send to {recipient} through {sender}: {message}"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}
    
@mcp.tool(name="broadcast_notification", description="Broadcast the notification from sender to more than one recipients")
def broadcast_notification(sender: str, recipients: list[str], subject: str, message: str):
    try:
        if not sender or not message or ( not recipients or len(recipients) == 0 ):
            raise Exception("BROADCAST_NOTIFICATION: sender, recipient and message are required fields!")
        if not sender.endswith("myorgemail.com"):
            raise Exception("BROADCAST_NOTIFICATION: sender is invalid!!")

        for recipient in recipients:
            if not recipient.endswith("myorgemail.com"):
                raise Exception("BROADCAST_NOTIFICATION: recipient is invalid!!")

        logger.info(f"""
Action: Broadcast Notification
Sender: {sender}
Recipient: {recipients}
Subject: {subject}
Message: {message}
""")
        return {"status": 200, "message": f"Notification has been send to {recipients} through {sender}: {message}"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}

@mcp.tool(name="schedule_notification", description="Schedule the notification from sender to more than one recipients based on the given cron expression")
def schedule_notification(sender: str, recipients: list[str], subject: str, message: str, cron_expression: str):
    try:
        if not sender or not message or ( not recipients or len(recipients) == 0 ):
            raise Exception("SCHEDULE_NOTIFICATION: sender, recipient, message and cron expression are required fields!")

        if not sender.endswith("myorgemail.com"):
                    raise Exception("SCHEDULE_NOTIFICATION: sender is invalid!!")
        
        for recipient in recipients:
            if not recipient.endswith("myorgemail.com"):
                raise Exception("SCHEDULE_NOTIFICATION: recipient is invalid!!")

        logger.info(f"""
        Action: Scheduled Notification
        Sender: {sender}
        Recipient: {recipients}
        Subject: {subject}
        Message: {message}
        CRON Expression: {cron_expression}
        """)
        return {"status": 200, "message": f"Notification has been send to {recipients} through {sender}: {message}"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}

@mcp.tool(name="schedule_meeting", description="Schedule the meeting and notify from sender to more than one recipients based on the given cron expression")
def schedule_meeting(sender: str, recipients: list[str], subject: str, message: str, cron_expression: str):
    try:
        if not sender or not message or ( not recipients or len(recipients) == 0 ):
            raise Exception("SCHEDULE_NOTIFICATION: sender, recipient, message and cron expression are required fields!")

        if not sender.endswith("myorgemail.com"):
                    raise Exception("SCHEDULE_NOTIFICATION: sender is invalid!!")
        
        for recipient in recipients:
            if not recipient.endswith("myorgemail.com"):
                raise Exception("SCHEDULE_NOTIFICATION: recipient is invalid!!")

        logger.info(f"""
Action: Scheduled Meeting
Sender: {sender}
Recipient: {recipients}
Subject: {subject}
Message: {message}
CRON Expression: {cron_expression}
""")
        return {"status": 200, "message": f"Meeting Notification has been send to {recipients} through {sender}: {message}"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": 500, "message": f"Error: {e}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")