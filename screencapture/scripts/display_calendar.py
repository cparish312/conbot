import tzlocal
import tkinter as tk
from tkinter import messagebox, ttk
from tkcalendar import Calendar
import pandas as pd
from datetime import datetime

from zoneinfo import ZoneInfo
#
local_timezone = tzlocal.get_localzone()
video_timezone = ZoneInfo("UTC")

df = pd.read_csv('/Users/connorparish/code/conbot/screencapture/screenshot_data/videos_summary.csv')
df['start_datetime'] = pd.to_datetime(df['start_datetime'])
df['start_datetime'] = df['start_datetime'].apply(lambda x: x.replace(tzinfo=video_timezone).astimezone(local_timezone))
df['end_datetime'] = pd.to_datetime(df['end_datetime'])
df['end_datetime'] = df['end_datetime'].apply(lambda x: x.replace(tzinfo=video_timezone).astimezone(local_timezone))
df = df.sort_values(by='start_datetime')

# Function to highlight event days on the calendar
def highlight_dates(cal, dataframe):
    for date in pd.date_range(dataframe['start_datetime'].dt.date.min(), dataframe['end_datetime'].dt.date.max()):
        events = dataframe[(dataframe['start_datetime'].dt.date <= date.date()) & (dataframe['end_datetime'].dt.date >= date.date())]
        if not events.empty:
            cal.calevent_create(date, 'Event', 'event')

def create_event_button(top_level, event_detail, row):
    """Creates an interactive button for an event."""
    button_text = f"Start: {row['start_datetime'].time().strftime('%H')} - End: {row['end_datetime'].time().strftime('%H')}"
    event_button = tk.Button(top_level, text=button_text, command=lambda: show_event_detail(event_detail))
    event_button.pack(fill='x')

def show_event_detail(event_detail):
    """Action to perform when an event button is clicked."""
    # Perform your desired action. For example, display event details.
    # This can be extended to show details in a new pop-up, edit event, etc.
    messagebox.showinfo("Event Detail", event_detail)

# Function to show event details
def show_event_details(date):
    events = df[(df['start_datetime'].dt.date <= date) & (df['end_datetime'].dt.date >= date)]

    events = events.sort_values(by='start_datetime') # Should already by done
    if not events.empty:
        # Create a new top-level window
        event_window = tk.Toplevel()
        event_window.title(f"Events on {date.strftime('%Y-%m-%d')}")

        # Group events by hour and take the first event of each hour
        events_grouped_by_hour = events.groupby(events['start_datetime'].dt.hour, as_index=False).first().reset_index()

        # For each event, create an interactive button
        for _, row in events_grouped_by_hour.iterrows():
            event_time_range = f"{row['start_datetime'].time().strftime('%H:%M')} - {row['end_datetime'].time().strftime('%H:%M')}"
            create_event_button(event_window, event_time_range, row)
    else:
        # If no events, you can either show a message or do nothing
        messagebox.showinfo("Event Details", "No events on this day.")

# Creating the main application window
root = tk.Tk()
root.title("Calendar Events")

# Styling the calendar to change the color of the current day
style = ttk.Style(root)
style.theme_use('clam')

style.configure('my.Calendar.Calendar', background='white', foreground='black', bordercolor='white', headersbackground='lightgrey', headersforeground='black', selectbackground='blue', selectforeground='white', normalbackground='white', normalforeground='black', weekendbackground='white', weekendforeground='black', othermonthforeground='grey', othermonthbackground='white', othermonthweforeground='grey', othermonthwebackground='white')

# Creating the calendar
cal = Calendar(root, selectmode='day', style='my.Calendar.Calendar')
cal.pack(pady=20)
highlight_dates(cal, df)

# Adding the selection event
cal.bind("<<CalendarSelected>>", lambda e: show_event_details(cal.selection_get()))

root.mainloop()