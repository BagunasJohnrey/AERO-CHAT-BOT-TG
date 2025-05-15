import os
import logging
import requests
import json
import openmeteo_requests
import requests_cache
import re
import asyncio
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
    """Improved weather service with better error handling and caching"""
    
    @staticmethod
    async def get_full_location(city: str) -> str:
        """Get clean location string with AI assistance"""
        try:
            prompt = f"Return ONLY the location in format 'City, Country'. Be concise. Input: {city}"
            system_msg = "You are a location formatting assistant. Return only the properly formatted location."
            
            location = await AIService.fetch_ai_response(
                prompt=prompt,
                system_message=system_msg,
                max_tokens=50,
                model="gpt-4o"
            )
            
            # Clean up the response
            location = location.strip().replace('"', '').replace("'", "")
            if location.lower().startswith("input:"):
                location = location[6:].strip()
                
            return location if location else city
            
        except Exception as e:
            logger.error(f"Location formatting error: {str(e)}")
            return city
    
    @staticmethod
    async def get_coordinates(city: str) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates with multiple fallback methods"""
        try:
            clean_city = clean_input(city)
            
            # Try with Open-Meteo first
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={clean_city}&count=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                result = data['results'][0]
                return result['latitude'], result['longitude']
                
            # Fallback to Nominatim if Open-Meteo fails
            url = f"https://nominatim.openstreetmap.org/search?q={clean_city}&format=json&limit=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, list):
                return float(data[0]['lat']), float(data[0]['lon'])
                
            return None, None
            
        except Exception as e:
            logger.error(f"Geocoding error for '{city}': {str(e)}")
            return None, None
    
    @staticmethod
    def fetch_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetch weather data with better error handling"""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": ["temperature_2m", "relative_humidity_2m", "precipitation", 
                            "weather_code", "surface_pressure", "wind_speed_10m", 
                            "wind_direction_10m", "uv_index"],
                "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min",
                         "precipitation_sum", "uv_index_max"],
                "timezone": "auto"
            }
            
            response = openmeteo.weather_api(url, params=params)[0]
            current = response.Current()
            daily = response.Daily()
            
            return {
                "current": {
                    "temp": current.Variables(0).Value(),
                    "humidity": current.Variables(1).Value(),
                    "precip": current.Variables(2).Value(),
                    "weather_code": current.Variables(3).Value(),
                    "pressure": current.Variables(4).Value(),
                    "wind_speed": current.Variables(5).Value(),
                    "wind_dir": current.Variables(6).Value(),
                    "uv": current.Variables(7).Value()
                },
                "daily": {
                    "weather_code": daily.Variables(0).ValuesAsNumpy(),
                    "temp_max": daily.Variables(1).ValuesAsNumpy(),
                    "temp_min": daily.Variables(2).ValuesAsNumpy(),
                    "precip_sum": daily.Variables(3).ValuesAsNumpy(),
                    "uv_max": daily.Variables(4).ValuesAsNumpy()
                },
                "timezone": response.Timezone()
            }
            
        except Exception as e:
            logger.error(f"Weather fetch error: {str(e)}")
            return None
    
    @staticmethod
    def get_weather_description(code: int) -> str:
        """Convert weather code to description"""
        codes = {
            0: "☀️ Clear sky",
            1: "🌤️ Mainly clear",
            2: "⛅ Partly cloudy",
            3: "☁️ Overcast",
            45: "🌫️ Fog",
            48: "❄️ Depositing rime fog",
            51: "🌧️ Light drizzle",
            53: "🌧️ Moderate drizzle",
            55: "🌧️ Dense drizzle",
            56: "🌧️ Freezing drizzle",
            57: "🌧️ Dense freezing drizzle",
            61: "🌧️ Slight rain",
            63: "🌧️ Moderate rain",
            65: "🌧️ Heavy rain",
            66: "🌨️ Freezing rain",
            67: "🌨️ Heavy freezing rain",
            71: "❄️ Slight snow",
            73: "❄️ Moderate snow",
            75: "❄️ Heavy snow",
            77: "❄️ Snow grains",
            80: "🌧️ Slight rain showers",
            81: "🌧️ Moderate rain showers",
            82: "🌧️ Violent rain showers",
            85: "❄️ Slight snow showers",
            86: "❄️ Heavy snow showers",
            95: "⛈️ Thunderstorm",
            96: "⛈️ Thunderstorm with hail",
            99: "⛈️ Heavy thunderstorm with hail"
        }
        return codes.get(code, "Unknown weather conditions")
    
    @staticmethod
    def get_uv_level(uv: float) -> Tuple[str, str]:
        """Get UV level and protection advice"""
        uv = float(uv)
        if uv < 3:
            return "Low", "🟢 No protection needed"
        elif uv < 6:
            return "Moderate", "🟡 Wear sunscreen and a hat"
        elif uv < 8:
            return "High", "🟠 Protection required - seek shade during midday"
        elif uv < 11:
            return "Very High", "🔴 Extra protection needed - avoid sun exposure"
        else:
            return "Extreme", "🚨 Avoid being outside during midday"

# Handlers
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

async def weather_location_handler(update: Update, context: CallbackContext) -> int:
    """Handle weather location input with option to get weather again"""
    if update.message:
        city = clean_input(update.message.text)
        
        if not validate_city_name(city):
            await update.message.reply_text(
                "⚠️ Please enter a valid city name (max 50 characters).",
                reply_markup=back_button()
            )
            return WEATHER_LOCATION
            
        await show_typing(context, update.message.chat_id)
        
        # Get location details
        full_location = await WeatherService.get_full_location(city)
        lat, lon = await WeatherService.get_coordinates(full_location)
        
        if not lat or not lon:
            await update.message.reply_text(
                f"❌ Couldn't find weather data for '{full_location}'. Try a nearby city.",
                reply_markup=back_button()
            )
            return WEATHER_LOCATION
            
        # Get weather data
        weather = WeatherService.fetch_weather(lat, lon)
        if not weather:
            await update.message.reply_text(
                f"⚠️ Weather service unavailable for {full_location}.",
                reply_markup=back_button()
            )
            return WEATHER_LOCATION
            
        # Format current weather
        current = weather['current']
        weather_desc = WeatherService.get_weather_description(current['weather_code'])
        uv_level, uv_advice = WeatherService.get_uv_level(current['uv'])
        
        weather_msg = (
            f"🌦️ *Weather for {full_location}*\n\n"
            f"{weather_desc}\n"
            f"🌡️ Temperature: *{current['temp']:.1f}°C*\n"
            f"💧 Humidity: *{current['humidity']}%*\n"
            f"🌬️ Wind: *{current['wind_speed']:.2f} km/h* from *{current['wind_dir']:.2f}°*\n"
            f"☔ Precipitation: *{current['precip']:.2f} mm*\n"
            f"☀️ UV Index: *{current['uv']:.1f} ({uv_level})*\n"
            f"{uv_advice}\n\n"
            f"Select an option below:"
        )
        
        await update.message.reply_text(
            weather_msg,
            parse_mode="Markdown",
            reply_markup=weather_menu()  # Now shows both weather again and back buttons
        )
        return MAIN_MENU
        
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
    """Get natural disaster events from AI"""
    prompt = (
        f"List 3 recent or upcoming natural disaster events (typhoons, earthquakes, floods, etc.) affecting {city} with dates and brief descriptions, maximum 200 words."
        if city else
        "List 3 recent or upcoming major natural disaster events (typhoons, earthquakes, floods, etc.) worldwide with dates and brief descriptions, maximum 200 words."
    )
    return "⚠️ *Natural Disaster Events* ⚠️\n\n" + await AIService.fetch_ai_response(
        prompt=prompt,
        system_message="You're a disaster response expert. Provide recent or upcoming natural disaster events with dates and impacts in clear bullet points, maximum 200 words. Focus on typhoons, earthquakes, floods and other natural disasters.",
        max_tokens=1200
    )

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
        "summary": "Mandates proper waste segregation, recycling, and disposal.",
        "irr": "DENR Administrative Order No. 2001-34",
        "penalty": "Fines from ₱300 to ₱1,000 for individuals",
        "imprisonment": "1 to 15 days community service",
        "link": "https://emb.gov.ph/wp-content/uploads/2015/09/RA-9003.pdf"
    },
    'law_water': {
        "title": "RA 9275 – Philippine Clean Water Act (2004)",
        "summary": "Protects water bodies from pollution from land-based sources.",
        "irr": "DENR Administrative Order No. 2005-10",
        "penalty": "Fines up to ₱200,000/day per violation",
        "imprisonment": "Up to 10 years",
        "link": "https://emb.gov.ph/wp-content/uploads/2015/09/RA-9275.pdf"
    },
    'law_air': {
        "title": "RA 8749 – Philippine Clean Air Act (1999)",
        "summary": "Aims to achieve and maintain clean air that meets national air quality standards.",
        "irr": "DENR Administrative Order No. 2000-81",
        "penalty": "Fines up to ₱100,000/day per violation",
        "imprisonment": "Up to 6 years",
        "link": "https://emb.gov.ph/wp-content/uploads/2015/09/RA-8749.pdf"
    },
    'law_climate': {
        "title": "RA 9729 – Climate Change Act (2009)",
        "summary": "Mainstreams climate change into government policy formulations.",
        "irr": "DENR Administrative Order No. 2010-01",
        "penalty": "As specified in respective provisions",
        "imprisonment": "As specified in respective provisions",
        "link": "https://climate.gov.ph/files/RA-9729.pdf"
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
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],  # Added start as fallback
        allow_reentry=True  # Allow users to re-enter the conversation
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