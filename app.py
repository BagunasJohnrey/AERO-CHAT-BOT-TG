import os
import logging
import requests
import json
import openmeteo_requests
import requests_cache
import re
import asyncio
import urllib.request
import json
from datetime import datetime, timedelta
from retry_requests import retry
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    CallbackContext,
    ContextTypes,
    ConversationHandler,
)
from dotenv import load_dotenv
from typing import Tuple, Optional, Dict, Any, List

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("aerobot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config
CONFIG = {
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN"),
    "WEATHER_API_CACHE_EXPIRE": 1800,  # 30 minutes
    "MAX_CITY_LENGTH": 50,
    "MAX_QUESTION_LENGTH": 200,
}

# States for conversation handler
MAIN_MENU, WEATHER_LOCATION, ASK_QUESTION, EVENTS_LOCATION, WATER_TIPS_LOCATION = range(5)

# Open-Meteo client
# Setup cached session
cache_session = requests_cache.CachedSession(
    '.cache',
    expire_after=CONFIG["WEATHER_API_CACHE_EXPIRE"],
    backend='sqlite'
)

# Create retry session with minimal configuration
retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

# Create OpenMeteo client
openmeteo = openmeteo_requests.Client(session=retry_session)

# Helper Functions
async def show_typing(context: CallbackContext, chat_id: int, duration: float = 1.0):
    """Show typing indicator for a duration"""
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    await asyncio.sleep(duration)  # Simulate processing time

def clean_input(text: str) -> str:
    """Clean user input by removing excessive whitespace and special characters"""
    text = re.sub(r'[^\w\s,.!?-]', '', text.strip())
    return re.sub(r'\s+', ' ', text)

def validate_city_name(city: str) -> bool:
    """Validate city name input"""
    if len(city) > CONFIG["MAX_CITY_LENGTH"]:
        return False
    if not re.match(r'^[\w\s\-,.]+$', city):
        return False
    return True

def validate_question(question: str) -> bool:
    """Validate question input"""
    if len(question) > CONFIG["MAX_QUESTION_LENGTH"]:
        return False
    if not question.strip():
        return False
    return True

def rate_limit_user(user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Simple rate limiting for users"""
    now = datetime.now()
    last_request = context.user_data.get('last_request')
    
    if last_request and (now - last_request) < timedelta(seconds=5):
        return True
    
    context.user_data['last_request'] = now
    return False

# Keyboards
def main_menu() -> InlineKeyboardMarkup:
    """Generate main menu keyboard"""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🌦️ Weather", callback_data='weather'),
            InlineKeyboardButton("❓ Ask Question", callback_data='ask')
        ],
        [
            InlineKeyboardButton("🌱 Eco Tips", callback_data='tips'),
            InlineKeyboardButton("📅 Events", callback_data='events')
        ],
        [
            InlineKeyboardButton("💧 Water Tips", callback_data='water'),
            InlineKeyboardButton("⚠️ Disaster Prep", callback_data='disaster')
        ],
        [
            InlineKeyboardButton("📜 Climate Laws", callback_data='laws'),
            InlineKeyboardButton("ℹ️ About", callback_data='about')
        ]
    ])

def weather_menu() -> InlineKeyboardMarkup:
    """Generate weather options keyboard with both back and weather again options"""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🌦️ Get Weather Again", callback_data='weather'),
            InlineKeyboardButton("🔙 Back", callback_data='back')
        ]
    ])

def disaster_menu() -> InlineKeyboardMarkup:
    """Generate disaster preparedness menu"""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔥 Wildfires", callback_data='prep_wildfire'),
            InlineKeyboardButton("🌀 Typhoons", callback_data='prep_typhoon')
        ],
        [
            InlineKeyboardButton("🌊 Floods", callback_data='prep_flood'),
            InlineKeyboardButton("🌋 Earthquakes", callback_data='prep_earthquake')
        ],
        [
            InlineKeyboardButton("🌡️ Heatwaves", callback_data='prep_heatwave'),
            InlineKeyboardButton("🌫️ Smog", callback_data='prep_smog')
        ],
        [InlineKeyboardButton("🔙 Back", callback_data='back')]
    ])
    
def laws_menu() -> InlineKeyboardMarkup:
    """Generate climate laws menu"""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Clean Air Act", callback_data='law_air'),
            InlineKeyboardButton("Clean Water Act", callback_data='law_water')
        ],
        [
            InlineKeyboardButton("Waste Management", callback_data='law_waste'),
            InlineKeyboardButton("Climate Change", callback_data='law_climate')
        ],
        [InlineKeyboardButton("🔙 Back", callback_data='back')]
    ])

def back_button() -> InlineKeyboardMarkup:
    """Simple back button"""
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data='back')]])

class AIService:
    """Wrapper class for AI services with improved error handling and caching"""

    @staticmethod
    async def fetch_ai_response(
        prompt: str,
        system_message: str = "",
        max_tokens: int = 1000,
        model: str = "DeepSeek-R1"
    ) -> str:
        """Fetch response from DeepSeek-R1 model via BetaDash API"""
        try:
            full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
            full_prompt = full_prompt.strip()

            url = "https://betadash-api-swordslush-production.up.railway.app/Deepseek-R1"
            params = {"ask": full_prompt}
            headers = {"Content-Type": "application/json"}

            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            content = data.get("response", "⚠️ No response from DeepSeek API.")

            # Format the response for Telegram (optional)
            content = re.sub(r'\*\*(.*?)\*\*', r'*\1*', content)
            content = re.sub(r'\n{3,}', '\n\n', content)

            if len(content) > max_tokens:
                content = content[:max_tokens] + "..."

            return content

        except requests.exceptions.Timeout:
            logger.warning("DeepSeek API request timed out")
            return "⚠️ Aero Bot service is taking too long to respond. Please try again later."
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API request failed: {str(e)}")
            return "⚠️ Aero Bot service is currently unavailable. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected DeepSeek error: {str(e)}")
            return "⚠️ An unexpected error occurred. Please try again."

# Weather Service
class WeatherService:
    """Weather service using Kaiz Weather API"""
    
    @staticmethod
    async def get_weather_data(city: str) -> Optional[Dict[str, Any]]:
        """Fetch weather data from Kaiz Weather API"""
        try:
            clean_city = clean_input(city.replace(' ', '+'))  # Format for URL
            url = f"https://kaiz-apis.gleeze.com/api/weather?q={clean_city}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "0" not in data:
                return None
                
            return data["0"]
            
        except Exception as e:
            logger.error(f"Weather API error for '{city}': {str(e)}")
            return None
    
    @staticmethod
    def get_weather_description(skycode: str) -> str:
        """Convert skycode to description"""
        codes = {
            "0": "☀️ Clear sky",
            "1": "🌤️ Mostly sunny",
            "2": "⛅ Partly cloudy",
            "3": "☁️ Mostly cloudy",
            "4": "🌧️ Light rain",
            "5": "🌧️ Rain",
            "6": "🌧️ Heavy rain",
            "7": "⛈️ Thunderstorm",
            "8": "🌨️ Snow",
            "9": "🌫️ Fog",
            "34": "🌤️ Mostly sunny"
        }
        return codes.get(skycode, f"Unknown weather (code: {skycode})")
    
    @staticmethod
    def get_heat_advisory(temp: float, feelslike: float) -> Tuple[str, str]:
        """Get heat advisory based on temperature and feels-like"""
        temp_diff = float(feelslike) - float(temp)
        temp = float(temp)
        
        if temp >= 40:
            return "Extreme Heat Danger", "🔥 Extreme heat warning! Avoid outdoor activities and stay hydrated"
        elif temp >= 35:
            if temp_diff > 5:
                return "High Heat & Humidity", "🥵 Very hot and humid. Limit outdoor exposure"
            return "High Heat", "☀️ Very hot - stay in shade and drink water"
        elif temp >= 30:
            if temp_diff > 5:
                return "Hot & Humid", "😓 Hot and sticky - take frequent breaks in shade"
            return "Warm", "☀️ Warm weather - stay hydrated"
        else:
            return "Mild", "😌 Comfortable temperature"

    @staticmethod
    def format_simple_forecast(forecasts: List[Dict[str, Any]]) -> str:
        """Simplified forecast format with emojis"""
        forecast_lines = []
        for forecast in forecasts[:5]:
            day = forecast['shortday']
            conditions = forecast['skytextday']
            emoji = "☀️" if "sunny" in conditions.lower() else \
                    "🌧️" if "rain" in conditions.lower() else \
                    "⛅" if "cloud" in conditions.lower() else "🌤️"
            
            forecast_lines.append(
                f"{emoji} {day}: {forecast['high']}°C/{forecast['low']}°C "
                f"({conditions}, {forecast['precip']}% rain)"
            )
        return "\n".join(forecast_lines)

# Handlers
async def weather_location_handler(update: Update, context: CallbackContext) -> int:
    """Handle weather location input with 5-day forecast"""
    if update.message:
        city = clean_input(update.message.text)
        
        if not validate_city_name(city):
            await update.message.reply_text(
                "⚠️ Please enter a valid city name (max 50 characters).",
                reply_markup=back_button()
            )
            return WEATHER_LOCATION
            
        await show_typing(context, update.message.chat_id)
        
        # Get weather data from Kaiz API
        weather_data = await WeatherService.get_weather_data(city)
        
        if not weather_data:
            await update.message.reply_text(
                f"❌ Couldn't find weather data for '{city}'. Try a nearby city.",
                reply_markup=back_button()
            )
            return WEATHER_LOCATION
            
        # Store weather data
        context.user_data['weather_data'] = weather_data
            
        # Extract data from response
        location = weather_data["location"]["name"]
        current = weather_data["current"]
        forecasts = weather_data["forecast"]
        
        # Format current weather
        weather_msg = (
            f"🌤️ *Current Weather in {location}*\n"
            f"📅 {current['day']}, {current['date']}\n"
            f"⏰ {current['observationtime']}\n\n"
            f"{current['skytext']}\n"
            f"🌡️ Temperature: {current['temperature']}°C (Feels like {current['feelslike']}°C)\n"
            f"💧 Humidity: {current['humidity']}%\n"
            f"🌬️ Wind: {current['winddisplay']}\n\n"
        )
        
        # Add heat advisory
        advisory, advice = WeatherService.get_heat_advisory(
            current["temperature"], 
            current["feelslike"]
        )
        weather_msg += f"⚠️ *{advisory}*\n{advice}\n\n"
        
        # Replace the table section in weather_location_handler with:
        weather_msg += "🌤️ *5-Day Weather Forecast:*\n"
        weather_msg += WeatherService.format_simple_forecast(forecasts)
        
        # Create simplified keyboard options
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh Weather", callback_data='weather')],
            [InlineKeyboardButton("🔙 Main Menu", callback_data='back')]
        ]
        
        await update.message.reply_text(
            weather_msg,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return MAIN_MENU
        
    return MAIN_MENU

async def start(update: Update, context: CallbackContext) -> int:
    """Start command handler that works on all devices"""
    user = update.effective_user
    welcome_msg = (
        f"🌍 Hello {user.first_name}! Welcome to *AeroBot* 🌱\n\n"
        "I'm your climate and weather assistant. Here's what I can help with:\n"
        "• Real-time weather data and forecasts 🌦️\n"
        "• Climate change information and tips 🌱\n"
        "• Disaster preparedness guides ⚠️\n"
        "• Environmental laws and regulations 📜\n\n"
        "How can I assist you today?"
    )
    
    # Clear any existing conversation state
    if context.chat_data is not None:
        context.chat_data.clear()
    
    # Always send a fresh message with main menu
    await update.message.reply_text(
        welcome_msg,
        parse_mode="Markdown",
        reply_markup=main_menu()
    )
    return MAIN_MENU

async def main_menu_handler(update: Update, context: CallbackContext) -> int:
    """Handle main menu navigation - modified to preserve messages"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'back':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="🌍 Main Menu 🌱\n\nSelect an option:",
            reply_markup=main_menu()
        )
        return MAIN_MENU
        
    elif query.data == 'weather':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="🌇 Enter a city name for weather information:",
            reply_markup=back_button()
        )
        return WEATHER_LOCATION
        
    elif query.data == 'ask':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="🌡️ What climate-related question would you like to ask?",
            reply_markup=back_button()
        )
        return ASK_QUESTION
        
    elif query.data == 'tips':
        await show_typing(context, query.message.chat_id)
        tips = await get_eco_tips()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=tips,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 More Tips", callback_data='tips')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
            ])
        )
        return MAIN_MENU
        
    elif query.data == 'events':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="📍 Enter a city for local events or leave blank for global events:",
            reply_markup=back_button()
        )
        return EVENTS_LOCATION
        
    elif query.data == 'water':
        await show_typing(context, query.message.chat_id)
        tips = await get_water_tips()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=tips,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 More Water Tips", callback_data='water')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
            ])
        )
        return MAIN_MENU
        
    elif query.data == 'disaster':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="⚠️ Select disaster type for preparedness info:",
            reply_markup=disaster_menu()
        )
        return MAIN_MENU
        
    elif query.data == 'laws':
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="📜 Select a climate law to view details:",
            reply_markup=laws_menu()
        )
        return MAIN_MENU
        
    elif query.data == 'about':
        about_msg = (
            "*AeroBot* 🌱\n\n"
            "An advanced climate and weather assistant bot.\n\n"
            "Features:\n"
            "• Accurate weather data from multiple sources\n"
            "• Climate change information\n"
            "• Disaster preparedness guides\n"
            "• Environmental law database\n"
            "\n\nGroup 4 - Super Science\n"
            "\nDeveloped with ❤️ for the planet"
        )
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=about_msg,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
            ])
        )
        return MAIN_MENU
        
    elif query.data.startswith('prep_'):
        disaster_type = query.data.split('_')[1]
        await show_typing(context, query.message.chat_id)
        guide = await get_disaster_prep(disaster_type)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=guide,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚠️ More Disaster Prep", callback_data='disaster')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
            ])
        )
        return MAIN_MENU
        
    elif query.data in PH_LAWS:
        law = PH_LAWS[query.data]
        msg = (
            f"📘 *{law['title']}*\n\n"
            f"📝 *Summary:* {law['summary']}\n\n"
            f"📄 *Implementing Rules:* {law['irr']}\n\n"
            f"💸 *Fine:* {law['penalty']}\n\n"
            f"🕒 *Imprisonment:* {law['imprisonment']}\n\n"
            f"🔗 [Read the full law]({law['link']})"
        )
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=msg,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📜 More Laws", callback_data='laws')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
            ])
        )
        return MAIN_MENU
        
    return MAIN_MENU

# async def weather_location_handler(update: Update, context: CallbackContext) -> int:
#     """Handle weather location input with option to get weather again"""
#     if update.message:
#         city = clean_input(update.message.text)
        
#         if not validate_city_name(city):
#             await update.message.reply_text(
#                 "⚠️ Please enter a valid city name (max 50 characters).",
#                 reply_markup=back_button()
#             )
#             return WEATHER_LOCATION
            
#         await show_typing(context, update.message.chat_id)
        
#         # Get weather data from Kaiz API
#         weather_data = await WeatherService.get_weather_data(city)
        
#         if not weather_data:
#             await update.message.reply_text(
#                 f"❌ Couldn't find weather data for '{city}'. Try a nearby city.",
#                 reply_markup=back_button()
#             )
#             return WEATHER_LOCATION
            
#         # Store weather data for potential next day forecast request
#         context.user_data['weather_data'] = weather_data
            
#         # Extract data from response
#         location = weather_data["location"]["name"]
#         current = weather_data["current"]
#         today_forecast = weather_data["forecast"][0]
#         tomorrow_forecast = weather_data["forecast"][1] if len(weather_data["forecast"]) > 1 else None
        
#         # Format weather message
#         weather_msg = (
#             f"🌤️ *Weather for {location}*\n\n"
#             f"{current['skytext']}\n"
#             f"🌡️ *Temperature:* {current['temperature']}°C (Feels like {current['feelslike']}°C)\n"
#             f"💧 *Humidity:* {current['humidity']}%\n"
#             f"🌬️ *Wind:* {current['winddisplay']}\n\n"
#             f"📅 *Today's Forecast*\n"
#             f"⬆️ High: {today_forecast['high']}°C | ⬇️ Low: {today_forecast['low']}°C\n"
#             f"🌧️ Precipitation: {today_forecast['precip']}%\n"
#             f"☀️ Conditions: {today_forecast['skytextday']}\n\n"
#             f"⚠️ *Advisory:* Warm weather. Stay hydrated!"
#         )
        
#         # Create keyboard with options
#         keyboard = [
#             [
#                 InlineKeyboardButton("🌦️ Get Weather Again", callback_data='weather'),
#                 InlineKeyboardButton("📅 Tomorrow's Forecast", callback_data='tomorrow')
#             ],
#             [InlineKeyboardButton("🔙 Main Menu", callback_data='back')]
#         ]
        
#         await update.message.reply_text(
#             weather_msg,
#             parse_mode="Markdown",
#             reply_markup=InlineKeyboardMarkup(keyboard)
#         )
#         return MAIN_MENU
        
#     return MAIN_MENU

async def tomorrow_forecast_handler(update: Update, context: CallbackContext) -> int:
    """Handle request for tomorrow's forecast"""
    query = update.callback_query
    await query.answer()
    
    weather_data = context.user_data.get('weather_data')
    if not weather_data or len(weather_data["forecast"]) < 2:
        await query.edit_message_text(
            text="⚠️ Tomorrow's forecast not available. Please request weather again.",
            reply_markup=back_button()
        )
        return MAIN_MENU
    
    location = weather_data["location"]["name"]
    tomorrow = weather_data["forecast"][1]
    
    forecast_msg = (
        f"📅 *Detailed Tomorrow's Forecast for {location}*\n\n"
        f"📅 Date: {tomorrow['date']} ({tomorrow['day']})\n"
        f"⬆️ Maximum Temperature: {tomorrow['high']}°C\n"
        f"⬇️ Minimum Temperature: {tomorrow['low']}°C\n"
        f"🌧️ Precipitation Chance: {tomorrow['precip']}%\n"
        f"☀️ Expected Conditions: {tomorrow['skytextday']}\n\n"
        f"🧭 Recommendations:\n"
        f"- {'🌂 Carry an umbrella' if int(tomorrow['precip']) > 30 else 'No rain expected'}\n"
        f"- {'🧴 Apply sunscreen' if 'sunny' in tomorrow['skytextday'].lower() else ''}\n"
        f"- {'👕 Dress lightly' if int(tomorrow['high']) > 30 else '👔 Normal attire recommended'}"
    )
    
    keyboard = [
        [InlineKeyboardButton("🌦️ Current Weather", callback_data='weather')],
        [InlineKeyboardButton("🔙 Main Menu", callback_data='back')]
    ]
    
    await query.edit_message_text(
        text=forecast_msg,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MAIN_MENU

async def tomorrow_forecast_handler(update: Update, context: CallbackContext) -> int:
    """Handle request for tomorrow's forecast"""
    query = update.callback_query
    await query.answer()
    
    weather_data = context.user_data.get('weather_data')
    if not weather_data:
        await query.edit_message_text(
            text="⚠️ Weather data not available. Please request weather again.",
            reply_markup=back_button()
        )
        return MAIN_MENU
    
    if len(weather_data["forecast"]) < 2:
        await query.edit_message_text(
            text="⚠️ Tomorrow's forecast not available.",
            reply_markup=back_button()
        )
        return MAIN_MENU
    
    location = weather_data["location"]["name"]
    tomorrow = weather_data["forecast"][1]
    
    forecast_msg = (
        f"📅 *Tomorrow's Forecast for {location}*\n\n"
        f"⬆️ High: {tomorrow['high']}°C | ⬇️ Low: {tomorrow['low']}°C\n"
        f"🌧️ Precipitation: {tomorrow['precip']}%\n"
        f"☀️ Conditions: {tomorrow['skytextday']}\n\n"
        f"📅 Date: {tomorrow['date']} ({tomorrow['day']})"
    )
    
    keyboard = [
        [InlineKeyboardButton("🌦️ Get Weather Again", callback_data='weather')],
        [InlineKeyboardButton("🔙 Main Menu", callback_data='back')]
    ]
    
    await query.edit_message_text(
        text=forecast_msg,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MAIN_MENU

async def ask_question_handler(update: Update, context: CallbackContext) -> int:
    """Handle climate questions - modified to preserve messages"""
    if update.message:
        question = clean_input(update.message.text)
        
        if not validate_question(question):
            await update.message.reply_text(
                "⚠️ Please enter a valid question (max 200 characters).",
                reply_markup=back_button()
            )
            return ASK_QUESTION
            
        await show_typing(context, update.message.chat_id, 2.0)
        
        answer = await ask_ai(question)
        await update.message.reply_text(
            answer,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❓ Ask Another", callback_data='ask')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
            ])
        )
        return MAIN_MENU
        
    return MAIN_MENU

async def events_location_handler(update: Update, context: CallbackContext) -> int:
    """Handle events location input - modified to preserve messages"""
    if update.message:
        location = clean_input(update.message.text)
        
        if location and not validate_city_name(location):
            await update.message.reply_text(
                "⚠️ Please enter a valid city name (max 50 characters).",
                reply_markup=back_button()
            )
            return EVENTS_LOCATION
            
        await show_typing(context, update.message.chat_id, 2.0)
        
        events = await get_climate_events(location if location else None)
        await update.message.reply_text(
            events,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📅 More Events", callback_data='events')],
                [InlineKeyboardButton("🏠 Main Menu", callback_data='back')]
            ])
        )
        return MAIN_MENU
        
    return MAIN_MENU

async def cancel(update: Update, context: CallbackContext) -> int:
    """Cancel and end the conversation"""
    await update.message.reply_text(
        "🌱 Thank you for using AeroBot!",
        reply_markup=main_menu()
    )
    return ConversationHandler.END

# AI-based functions
async def get_eco_tips() -> str:
    """Get eco tips from AI"""
    return "🌿 *Eco Tips* 🌿\n\n" + await AIService.fetch_ai_response(
        prompt="Provide 5 practical eco-friendly tips with emojis maximum 200 words",
        system_message="You're an environmental expert. Provide actionable eco tips, maximum 200 words.",
        max_tokens=1000
    )

async def ask_ai(question: str) -> str:
    """Get answer to climate question from AI"""
    return await AIService.fetch_ai_response(
        prompt=question,
        system_message="You're a climate scientist. Provide accurate, concise answers to climate questions, maximum 200 words.",
        max_tokens=1500
    )

async def get_climate_events(city: Optional[str] = None) -> str:
    """Get climate-related news events from GNews API using urllib"""
    try:
        # Get API key from environment variables
        api_key = os.getenv("GNEWS_API_KEY") or "ebd3c240d560cee99713aac96e690a32"
        
        # Build base URL and query
        base_url = "https://gnews.io/api/v4/search"
        query = 'climate OR weather OR disaster OR flood OR typhoon OR earthquake'
        
        if city:
            query += f' AND {city}'
        
        url = f"{base_url}?q={urllib.parse.quote(query)}&lang=en&max=3&apikey={api_key}"
        
        # Make the API request
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data.get("articles", [])
            
            if not articles:
                return "🌍 No climate news found currently. Check back later for updates!"
            
            # Format the news articles
            events_msg = "🌦️ *Climate News Updates*\n\n"
            for article in articles[:3]:  # Limit to 3 articles
                title = article.get('title', 'No title')
                description = article.get('description', '')
                source = article.get('source', {}).get('name', 'Unknown source')
                published_at = article.get('publishedAt', '')
                url = article.get('url', '#')
                
                # Format date if available
                date_str = ""
                if published_at:
                    try:
                        date_obj = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                        date_str = date_obj.strftime("%b %d, %Y")
                    except ValueError:
                        date_str = published_at[:10]  # Just show YYYY-MM-DD if parsing fails
                
                events_msg += (
                    f"📰 *{title}*\n"
                    f"{description}\n"
                    f"📡 Source: {source}\n"
                )
                if date_str:
                    events_msg += f"📅 Date: {date_str}\n"
                events_msg += f"🔗 [Read more]({url})\n\n"
            
            return events_msg
            
    except urllib.error.URLError as e:
        logger.error(f"GNews API request failed: {str(e)}")
        return "⚠️ Could not fetch climate news. Please try again later."
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding API response: {str(e)}")
        return "⚠️ Error processing news data. Please try again."
    except Exception as e:
        logger.error(f"Error processing climate news: {str(e)}")
        return "⚠️ An error occurred while fetching climate news."

async def get_water_tips(region: Optional[str] = None) -> str:
    """Get water saving tips from AI"""
    prompt = f"Provide 5 water conservation tips for {region}." if region else "Provide 5 general water conservation tips, maximum 200 words."
    return "💧 *Water-Saving Tips* 💧\n\n" + await AIService.fetch_ai_response(
        prompt=prompt,
        system_message="You're a water conservation expert. Provide practical tips with emojis, maximum 200 words.",
        max_tokens=1000
    )

async def get_disaster_prep(disaster_type: str) -> str:
    """Get disaster preparedness guide from AI"""
    return f"⚠️ *{disaster_type.capitalize()} Preparedness* ⚠️\n\n" + await AIService.fetch_ai_response(
        prompt=f"Provide a 5-step preparedness guide for {disaster_type}, maximum 200 words.",
        system_message="You're a disaster preparedness expert. Provide clear, actionable steps with emojis, maximum 200 words.",
        max_tokens=1200
    )

# Philippine Environmental Laws
PH_LAWS = {
    'law_waste': {
        "title": "RA 9003 – Ecological Solid Waste Management Act (2000)",
        "summary": "Comprehensive legislation mandating proper segregation, recycling, composting, and disposal of solid waste through the establishment of Materials Recovery Facilities (MRFs) in every barangay.",
        "key_provisions": [
            "Mandatory segregation at source (household level)",
            "Prohibition of open dumping and burning of waste",
            "Establishment of sanitary landfills",
            "Extended Producer Responsibility (EPR) for manufacturers"
        ],
        "scope": "Applies to all waste generators including households, institutions, commercial establishments, and industries",
        "irr": "DENR Administrative Order No. 2001-34",
        "penalty": "Fines from ₱300 to ₱1,000 for individuals; ₱5,000 to ₱200,000 for establishments",
        "imprisonment": "1 to 15 days community service for individuals; 1-6 years for serious violations",
        "enforcement": "Local government units (LGUs) with DENR oversight",
        "link": "https://emb.gov.ph/wp-content/uploads/2015/09/RA-9003.pdf"
    },
    'law_water': {
        "title": "RA 9275 – Philippine Clean Water Act (2004)",
        "summary": "Comprehensive water quality management framework protecting all water bodies from land-based pollution sources including industries, commercial establishments, and agricultural activities.",
        "key_provisions": [
            "Wastewater charge system",
            "Water quality management areas",
            "Prohibition on discharging untreated wastewater",
            "Mandatory wastewater treatment facilities"
        ],
        "scope": "Covers all water bodies: inland surface waters, ground water, coastal and marine waters",
        "irr": "DENR Administrative Order No. 2005-10",
        "penalty": "Fines up to ₱200,000/day per violation for serious offenses",
        "imprisonment": "Up to 10 years for willful violations",
        "enforcement": "DENR through Environmental Management Bureau (EMB)",
        "link": "https://emb.gov.ph/wp-content/uploads/2015/09/RA-9275.pdf"
    },
    'law_air': {
        "title": "RA 8749 – Philippine Clean Air Act (1999)",
        "summary": "National air quality management program that sets emission standards for mobile and stationary sources, and phases out ozone-depleting substances.",
        "key_provisions": [
            "Ban on leaded gasoline",
            "Vehicle emission testing program",
            "Industrial emission limits",
            "Smoke Belching Reduction Program"
        ],
        "scope": "Regulates all potential air pollution sources including vehicles, factories, power plants, and open burning",
        "irr": "DENR Administrative Order No. 2000-81",
        "penalty": "Fines up to ₱100,000/day per violation",
        "imprisonment": "Up to 6 years for gross violations",
        "enforcement": "DENR-EMB with LGU and DOTC/LTO coordination",
        "link": "https://emb.gov.ph/wp-content/uploads/2015/09/RA-8749.pdf"
    },
    'law_climate': {
        "title": "RA 9729 – Climate Change Act (2009)",
        "summary": "Establishes the Climate Change Commission and mandates the formulation of the National Climate Change Action Plan (NCCAP) to mainstream climate change in policy formulation.",
        "key_provisions": [
            "Created the Climate Change Commission",
            "National Framework Strategy on Climate Change",
            "Local Climate Change Action Plans (LCCAPs)",
            "People's Survival Fund for adaptation projects"
        ],
        "scope": "Whole-of-government approach covering mitigation and adaptation strategies",
        "irr": "DENR Administrative Order No. 2010-01",
        "penalty": "Non-compliance may result in administrative sanctions (Sec. 19) including:\n• Suspension or cancellation of permits\n• Withholding of government benefits\n• Other appropriate penalties under existing laws",
        "imprisonment": "Violations may be subject to penalties under:\n• Revised Penal Code\n• Other relevant environmental laws\n• Implementing rules of specific provisions",
        "enforcement": "Climate Change Commission with all government agencies",
        "link": "https://climate.emb.gov.ph/?page_id=68"
    }
}

# Main function
def main() -> None:
    """Run the bot."""
    application = Application.builder().token(CONFIG["TELEGRAM_TOKEN"]).build()

    # Add conversation handler with the states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [CallbackQueryHandler(main_menu_handler)],
            WEATHER_LOCATION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, weather_location_handler),
                CallbackQueryHandler(main_menu_handler, pattern='^back$')
            ],
            ASK_QUESTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_question_handler),
                CallbackQueryHandler(main_menu_handler, pattern='^back$')
            ],
            EVENTS_LOCATION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, events_location_handler),
                CallbackQueryHandler(main_menu_handler, pattern='^back$')
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
        allow_reentry=True
    )

    application.add_handler(conv_handler)
    
    # Log all errors
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    logger.info("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

async def error_handler(update: Update, context: CallbackContext) -> None:
    """Log errors and send a message to the user."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ An unexpected error occurred. Please try again later.",
            reply_markup=main_menu()
        )

if __name__ == "__main__":
    main()
