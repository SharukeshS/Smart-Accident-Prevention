from __future__ import annotations

import os, json, pickle, time, math, hashlib, random
from datetime import datetime
from typing import Optional, Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
OPENWEATHER_KEY   = os.getenv("OPENWEATHER_API_KEY", "")
TOMTOM_KEY        = os.getenv("TOMTOM_API_KEY", "")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
WEATHER_CACHE_TTL = 600
ROUTE_CACHE_TTL   = 300

WEATHER_MAP = {"Clear":0,"Drizzle":1,"Light Rain":2,"Heavy Rain":3,"Fog":4,"Thunderstorm":5}
ROAD_MAP    = {"Highway":0,"Arterial":1,"Urban":2,"Coastal":3,"Industrial":4,"Flyover":5}
TIME_MAP    = {"Early Morning":0,"Morning Peak":1,"Afternoon":2,"Night":3,"Evening Peak":4}

CHENNAI_ZONES = {
    "Anna Salai":      {"lat":13.0637,"lng":80.2565,"base_risk":85},
    "Kathipara":       {"lat":13.0095,"lng":80.2105,"base_risk":92},
    "OMR IT Corridor": {"lat":12.9279,"lng":80.2211,"base_risk":62},
    "GST Road":        {"lat":12.9716,"lng":80.1999,"base_risk":75},
    "Adyar":           {"lat":13.0067,"lng":80.2571,"base_risk":58},
    "Koyambedu":       {"lat":13.0694,"lng":80.1948,"base_risk":70},
    "Velachery":       {"lat":13.0068,"lng":80.2209,"base_risk":55},
    "Manali":          {"lat":13.1666,"lng":80.2573,"base_risk":48},
    "Poonamallee":     {"lat":13.0458,"lng":80.1533,"base_risk":28},
    "ECR":             {"lat":12.9023,"lng":80.2527,"base_risk":18},
    "T Nagar":         {"lat":13.0418,"lng":80.2341,"base_risk":78},
    "Tambaram":        {"lat":12.9229,"lng":80.1275,"base_risk":65},
    "Ambattur":        {"lat":13.1143,"lng":80.1548,"base_risk":55},
    "Porur":           {"lat":13.0358,"lng":80.1572,"base_risk":60},
    "Pallavaram":      {"lat":12.9675,"lng":80.1499,"base_risk":52},
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chennai TrafficAI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
def cache_get(key, ttl):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < ttl:
            return val
    return None
def cache_set(key, val):
    _cache[key] = (val, time.time())

# ── Model ─────────────────────────────────────────────────────────────────────
_model_bundle = None
def get_model():
    global _model_bundle
    if _model_bundle is None:
        pkl = os.path.join(MODELS_DIR, "accident_model.pkl")
        if not os.path.exists(pkl):
            raise HTTPException(503, "Model not trained.")
        with open(pkl, "rb") as f:
            _model_bundle = pickle.load(f)
    return _model_bundle

# ── Pydantic ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    latitude:          float = 13.0637
    longitude:         float = 80.2565
    hour:              int   = Field(default_factory=lambda: datetime.now().hour)
    is_weekend:        int   = Field(default_factory=lambda: int(datetime.now().weekday()>=5))
    temperature_c:     float = 30.0
    humidity_pct:      float = 70.0
    wind_speed_kmh:    float = 15.0
    visibility_km:     float = 8.0
    rainfall_mm:       float = 0.0
    congestion_pct:    float = 50.0
    speed_kmh:         float = 30.0
    is_junction:       int   = 0
    is_school_zone:    int   = 0
    flood_risk:        int   = 0
    weather_condition: str   = "Clear"
    road_type:         str   = "Urban"
    time_period:       str   = "Afternoon"

class FullAlertRequest(BaseModel):
    latitude:    float
    longitude:   float
    destination: Optional[str] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def hour_to_period(h):
    if 8<=h<=10:  return "Morning Peak"
    if 17<=h<=20: return "Evening Peak"
    if 23<=h or h<=4: return "Early Morning"
    if 11<=h<=14: return "Afternoon"
    return "Night"

def nearest_zone(lat, lng):
    best, bd = "Anna Salai", 1e9
    for name, d in CHENNAI_ZONES.items():
        dist = math.sqrt((lat-d["lat"])**2+(lng-d["lng"])**2)
        if dist < bd: bd, best = dist, name
    return best, CHENNAI_ZONES[best]

# ── ML Prediction ─────────────────────────────────────────────────────────────
def run_prediction(req: PredictRequest):
    bundle = get_model()
    xgb, rf, scaler = bundle["xgb"], bundle["rf"], bundle["scaler"]
    we = WEATHER_MAP.get(req.weather_condition, 0)
    re = ROAD_MAP.get(req.road_type, 2)
    te = TIME_MAP.get(req.time_period, 0)
    X = np.array([[req.hour, req.is_weekend, req.temperature_c, req.humidity_pct,
                   req.wind_speed_kmh, req.visibility_km, req.rainfall_mm,
                   req.congestion_pct, req.speed_kmh, req.is_junction,
                   req.is_school_zone, req.flood_risk, we, re, te]], dtype=np.float32)
    xgb_prob = float(xgb.predict_proba(X)[0][1])
    rf_prob  = float(rf.predict_proba(scaler.transform(X))[0][1])
    prob     = 0.55*xgb_prob + 0.45*rf_prob
    risk_level = "CRITICAL" if prob>0.75 else "HIGH" if prob>0.55 else "MODERATE" if prob>0.35 else "LOW"
    alerts = []
    if req.rainfall_mm > 15:    alerts.append("⚠️ Heavy rainfall — reduced braking distance")
    if req.visibility_km < 2:   alerts.append("🌫️ Low visibility — use hazard lights")
    if req.flood_risk:          alerts.append("🌊 Flood risk — avoid underpasses")
    if req.congestion_pct > 80: alerts.append("🚗 Severe congestion — consider alternate route")
    if req.is_junction:         alerts.append("🛑 Intersection zone — slow down")
    if 17<=req.hour<=20:        alerts.append("🕔 Peak hour — elevated collision risk")
    return {
        "accident_probability": round(prob,4),
        "accident_predicted":   int(prob>=0.45),
        "risk_score":           round(prob*100,1),
        "risk_level":           risk_level,
        "xgb_probability":      round(xgb_prob,4),
        "rf_probability":       round(rf_prob,4),
        "alerts":               alerts,
    }

# ── Weather ───────────────────────────────────────────────────────────────────
def fetch_weather_owm(lat, lng):
    key = f"weather_{round(lat,2)}_{round(lng,2)}"
    cached = cache_get(key, WEATHER_CACHE_TTL)
    if cached: return {**cached, "cached": True}
    if not OPENWEATHER_KEY: return _mock_weather()
    try:
        r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={OPENWEATHER_KEY}&units=metric", timeout=8)
        d = r.json()
        rain = d.get("rain",{})
        wind = d.get("wind",{})
        cmap = {"Thunderstorm":"Thunderstorm","Drizzle":"Drizzle","Rain":"Heavy Rain" if rain.get("1h",0)>5 else "Light Rain","Mist":"Fog","Fog":"Fog","Clear":"Clear","Clouds":"Clear"}
        cond = cmap.get(d["weather"][0]["main"],"Clear")
        result = {
            "temperature_c": round(d["main"]["temp"],1),
            "feels_like_c":  round(d["main"]["feels_like"],1),
            "humidity_pct":  d["main"]["humidity"],
            "wind_speed_kmh":round(wind.get("speed",0)*3.6,1),
            "wind_gust_kmh": round(wind.get("gust",0)*3.6,1),
            "visibility_km": round(d.get("visibility",10000)/1000,1),
            "rainfall_mm":   round(rain.get("1h",0),1),
            "weather_condition": cond,
            "weather_desc":  d["weather"][0]["description"].upper(),
            "weather_icon":  {"Thunderstorm":"⛈️","Heavy Rain":"🌧️","Light Rain":"🌦️","Fog":"🌫️","Clear":"☀️"}.get(cond,"⛅"),
            "pressure_hpa":  d["main"].get("pressure",1013),
            "cached": False, "source": "OpenWeatherMap",
        }
        result["traffic_advisory"] = _weather_advisory(result)
        cache_set(key, result)
        return result
    except Exception as e:
        return _mock_weather()

def _mock_weather():
    month = datetime.now().month
    rain_prob = 0.5 if 6<=month<=11 else 0.15
    if random.random() < rain_prob:
        cond = random.choice(["Heavy Rain","Light Rain","Thunderstorm"])
        rain, vis = round(random.uniform(5,35),1), round(random.uniform(0.8,4.0),1)
    else:
        cond, rain, vis = "Clear", 0.0, round(random.uniform(6,15),1)
    result = {
        "temperature_c": round(random.gauss(31,3),1), "feels_like_c": round(random.gauss(34,3),1),
        "humidity_pct": round(random.uniform(60,90),1), "wind_speed_kmh": round(random.uniform(5,25),1),
        "wind_gust_kmh": round(random.uniform(10,40),1), "visibility_km": vis,
        "rainfall_mm": rain, "weather_condition": cond, "weather_desc": cond.upper(),
        "weather_icon": {"Thunderstorm":"⛈️","Heavy Rain":"🌧️","Light Rain":"🌦️","Clear":"☀️"}.get(cond,"⛅"),
        "pressure_hpa": 1008, "cached": False, "source": "mock",
    }
    result["traffic_advisory"] = _weather_advisory(result)
    return result

def _weather_advisory(w):
    alerts, severity = [], "OK"
    if w["rainfall_mm"]>20:    alerts.append("⚠️ HEAVY RAIN — avoid underpasses"); severity="DANGER"
    elif w["rainfall_mm"]>5:   alerts.append("🌧️ MODERATE RAIN — slow down"); severity="WARN"
    if w["visibility_km"]<1:   alerts.append("🌫️ VERY LOW VISIBILITY — hazard lights"); severity="DANGER"
    elif w["visibility_km"]<3: alerts.append("🌫️ REDUCED VISIBILITY — caution"); severity=severity if severity!="OK" else "WARN"
    if w["wind_speed_kmh"]>30: alerts.append("💨 STRONG WINDS — two-wheelers caution"); severity=severity if severity!="OK" else "WARN"
    if w["weather_condition"]=="Thunderstorm": alerts.append("⛈️ THUNDERSTORM — avoid travel"); severity="DANGER"
    if not alerts: alerts.append("✅ CONDITIONS NORMAL — standard safety protocols")
    return {"alerts": alerts, "severity": severity}

# ── TomTom Routing ────────────────────────────────────────────────────────────
def _tomtom_geocode(place: str):
    """Geocode place name to (lat, lng) using TomTom"""
    try:
        from urllib.parse import quote
        q = quote(f"{place}, Chennai, Tamil Nadu, India")
        r = requests.get(
            f"https://api.tomtom.com/search/2/geocode/{q}.json",
            params={"key": TOMTOM_KEY, "countrySet": "IN", "limit": 1},
            timeout=8
        )
        pos = r.json()["results"][0]["position"]
        print(f"GEOCODE: {place} → {pos['lat']}, {pos['lon']}")
        return pos["lat"], pos["lon"]
    except Exception as e:
        print(f"GEOCODE ERROR: {e}")
        return None

def _tomtom_route(olat, olng, dlat, dlng, route_type="fastest"):
    """Fetch a route from TomTom Routing API"""
    r = requests.get(
        f"https://api.tomtom.com/routing/1/calculateRoute/{olat},{olng}:{dlat},{dlng}/json",
        params={
            "key": TOMTOM_KEY, "traffic": "true", "travelMode": "car",
            "routeType": route_type, "computeTravelTimeFor": "all",
            "routeRepresentation": "polyline",
        },
        timeout=10
    )
    print(f"TOMTOM {route_type}: {r.status_code}")
    return r.json()

def _extract_coords(data):
    try:
        pts = data["routes"][0]["legs"][0]["points"]
        return [[p["longitude"], p["latitude"]] for p in pts]
    except:
        return []

def fetch_routes(origin: str, destination: str) -> dict:
    print(f"FETCH_ROUTES: {origin} → {destination}")
    cache_key = f"route_{hashlib.md5((origin+destination).encode()).hexdigest()}"
    cached = cache_get(cache_key, ROUTE_CACHE_TTL)
    if cached: return {**cached, "cached": True}

    if not TOMTOM_KEY:
        return _mock_routes(origin, destination)

    try:
        # Geocode origin (if place name) or parse coordinates
        try:
            parts = origin.strip().split(",")
            olat, olng = float(parts[0]), float(parts[1])
        except:
            geo = _tomtom_geocode(origin)
            if not geo: return _mock_routes(origin, destination)
            olat, olng = geo
        dest_geo = _tomtom_geocode(destination)

        # Geocode destination
        dest_geo = _tomtom_geocode(destination)
        if not dest_geo: return _mock_routes(origin, destination)
        dlat, dlng = dest_geo

        # Fetch 3 route types
        configs = [
            ("fastest", f"{destination} — Fastest Route"),
            ("shortest", f"{destination} — Shortest Route"),
            ("eco",      f"{destination} — Eco Route"),
        ]

        routes = []
        for rtype, rname in configs:
            try:
                data = _tomtom_route(olat, olng, dlat, dlng, rtype)
                if "routes" not in data or not data["routes"]: continue
                summ = data["routes"][0]["summary"]
                dist_km   = round(summ["lengthInMeters"] / 1000, 1)
                dur_min   = round(summ["travelTimeInSeconds"] / 60)
                delay_min = round(summ.get("trafficDelayInSeconds", 0) / 60)
                no_traffic = round(summ.get("noTrafficTravelTimeInSeconds", summ["travelTimeInSeconds"]) / 60)
                cong = min(100, int((delay_min / max(no_traffic,1)) * 200)) if no_traffic > 0 else 0
                status = "HEAVY" if cong>60 else "MODERATE" if cong>30 else "CLEAR"
                coords = _extract_coords(data)
                print(f"  {rtype}: {dist_km}km, {dur_min}min, delay:{delay_min}min, cong:{cong}%")
                routes.append({
                    "rank": len(routes)+1, "summary": rname,
                    "distance_km": dist_km, "duration_min": no_traffic,
                    "duration_traffic_min": dur_min, "delay_min": delay_min,
                    "congestion_pct": cong, "traffic_status": status,
                    "coordinates": coords, "polyline": "",
                    "warnings": [], "is_recommended": False,
                })
            except Exception as e:
                print(f"  {rtype} ERROR: {e}")

        if not routes: return _mock_routes(origin, destination)
        routes.sort(key=lambda x: x["duration_traffic_min"])
        for i, rt in enumerate(routes):
            rt["rank"] = i+1
            rt["is_recommended"] = (i==0)

        result = {"routes": routes, "recommended": routes[0],
                  "origin": origin, "destination": destination,
                  "cached": False, "source": "TomTom"}
        cache_set(cache_key, result)
        return result

    except Exception as e:
        print(f"FETCH_ROUTES ERROR: {e}")
        return _mock_routes(origin, destination)

def _mock_routes(origin, destination):
    routes = [
        {"rank":1,"summary":f"{destination} — Fastest","distance_km":round(random.uniform(4,15),1),
         "duration_min":round(random.uniform(12,35)),"duration_traffic_min":round(random.uniform(18,55)),
         "delay_min":round(random.uniform(5,20)),"congestion_pct":random.randint(20,85),
         "traffic_status":random.choice(["CLEAR","MODERATE","HEAVY"]),"polyline":"","coordinates":[],"warnings":[],"is_recommended":True},
        {"rank":2,"summary":f"{destination} — Alternate","distance_km":round(random.uniform(6,18),1),
         "duration_min":round(random.uniform(15,45)),"duration_traffic_min":round(random.uniform(20,50)),
         "delay_min":round(random.uniform(3,15)),"congestion_pct":random.randint(10,60),
         "traffic_status":random.choice(["CLEAR","MODERATE"]),"polyline":"","coordinates":[],"warnings":[],"is_recommended":False},
    ]
    return {"routes":routes,"recommended":routes[0],"origin":origin,"destination":destination,"cached":False,"source":"mock"}

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    idx = os.path.join(STATIC_DIR, "index.html")
    print(f"Looking for index.html at: {idx}")
    print(f"File exists: {os.path.exists(idx)}")
    return FileResponse(idx) if os.path.exists(idx) else {"message":"Chennai TrafficAI API v2.0","docs":"/docs"}
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_ready": os.path.exists(os.path.join(MODELS_DIR,"accident_model.pkl")),
        "weather_api": bool(OPENWEATHER_KEY),
        "maps_api":    bool(TOMTOM_KEY),
        "timestamp":   datetime.now().isoformat(),
    }

@app.post("/api/predict/accident")
def predict_accident(req: PredictRequest):
    if req.time_period == "Afternoon": req.time_period = hour_to_period(req.hour)
    return run_prediction(req)

@app.post("/api/predict/batch")
def predict_batch():
    hour = datetime.now().hour
    period = hour_to_period(hour)
    is_wk = int(datetime.now().weekday()>=5)
    results = []
    for name, z in CHENNAI_ZONES.items():
        req = PredictRequest(latitude=z["lat"],longitude=z["lng"],hour=hour,
                             is_weekend=is_wk,congestion_pct=float(z["base_risk"]),time_period=period)
        pred = run_prediction(req)
        results.append({"zone":name,"latitude":z["lat"],"longitude":z["lng"],**pred})
    results.sort(key=lambda x: x["accident_probability"],reverse=True)
    return {"zones":results,"timestamp":datetime.now().isoformat()}

@app.get("/api/weather")
def get_weather(lat: float = Query(13.0827), lng: float = Query(80.2707)):
    return fetch_weather_owm(lat, lng)

@app.get("/api/routes")
def get_routes(
    origin:      str = Query(...),
    destination: str = Query(...),
):
    return fetch_routes(origin, destination)

@app.post("/api/alerts/full")
def full_alert(req: FullAlertRequest):
    lat, lng = req.latitude, req.longitude
    zone_name, zone_data = nearest_zone(lat, lng)
    weather = fetch_weather_owm(lat, lng)
    hour = datetime.now().hour
    pred_req = PredictRequest(
        latitude=lat, longitude=lng, hour=hour,
        is_weekend=int(datetime.now().weekday()>=5),
        temperature_c=weather.get("temperature_c",30),
        humidity_pct=weather.get("humidity_pct",70),
        wind_speed_kmh=weather.get("wind_speed_kmh",15),
        visibility_km=weather.get("visibility_km",8),
        rainfall_mm=weather.get("rainfall_mm",0),
        congestion_pct=float(zone_data["base_risk"]),
        weather_condition=weather.get("weather_condition","Clear"),
        flood_risk=int(weather.get("rainfall_mm",0)>10),
        time_period=hour_to_period(hour),
    )
    prediction = run_prediction(pred_req)
    routes = fetch_routes(f"{lat},{lng}", req.destination) if req.destination else None
    all_alerts = weather.get("traffic_advisory",{}).get("alerts",[]) + prediction.get("alerts",[])
    seen, deduped = set(), []
    for a in all_alerts:
        k = a[:30]
        if k not in seen: seen.add(k); deduped.append(a)
    severity = weather.get("traffic_advisory",{}).get("severity","OK")
    if prediction["risk_level"] in ("CRITICAL","HIGH") and severity!="DANGER": severity="WARN"
    return {"zone":zone_name,"latitude":lat,"longitude":lng,"weather":weather,
            "prediction":prediction,"routes":routes,"alerts":deduped,"severity":severity,
            "timestamp":datetime.now().isoformat()}

@app.get("/api/zones")
def get_zones(): return {"zones": CHENNAI_ZONES}

@app.get("/api/model/meta")
def model_meta():
    mp = os.path.join(MODELS_DIR,"model_meta.json")
    if not os.path.exists(mp): raise HTTPException(404,"Model not trained yet.")
    with open(mp) as f: return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
