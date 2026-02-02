# 🏥 Medical Chatbot Monitoring Dashboard

**Live App URL:** `https://medical-chatbot.onrender.com`  
**LangSmith Project:** `medical-chatbot-render`  
**Last Updated:** [Auto-update this date]

---

## 🔗 Quick Access Links

### 🎯 Live Monitoring
| Dashboard | Purpose | Link |
|-----------|---------|------|
| **All Traces** | See every chat conversation | [📊 Open](https://smith.langchain.com/o/medical-chatbot-render) |
| **Errors Only** | Find and fix problems | [❌ Open](https://smith.langchain.com/o/medical-chatbot-render/filters/status%3D%22ERROR%22) |
| **Slow Queries** (>5s) | Optimize performance | [🐌 Open](https://smith.langchain.com/o/medical-chatbot-render/filters/duration%3E5000) |
| **High Cost** (>$0.10) | Monitor expenses | [💰 Open](https://smith.langchain.com/o/medical-chatbot-render/filters/estimated_cost%3E0.1) |

### 🛠️ System Status
| Component | Status Check | Link |
|-----------|--------------|------|
| **App Health** | Is the bot running? | [🩺 Open](https://medical-chatbot.onrender.com/health) |
| **Debug Info** | Technical details | [🔧 Open](https://medical-chatbot.onrender.com/debug) |
| **LangSmith Test** | Test monitoring | [🧪 Open](https://medical-chatbot.onrender.com/langsmith-test) |
| **Render Logs** | Server logs | [📋 Open Render Dashboard](https://dashboard.render.com) |

---

## 📈 Daily Monitoring Checklist

### 🌅 Morning Check (9:00 AM)
1. **✅ Check health:** `https://medical-chatbot.onrender.com/health`
2. **✅ Review errors:** Check LangSmith for overnight errors
3. **✅ Check costs:** Ensure < $1 spent yesterday
4. **✅ Test response:** Send a test message to bot

### 🌇 Evening Check (6:00 PM)
1. **✅ Error count:** Should be < 5 for the day
2. **✅ Response time:** Should be < 3 seconds average
3. **✅ User count:** Check unique sessions
4. **✅ Save report:** Take screenshot of dashboard

---

## 🚨 Alert Triggers (When to Take Action)

| Alert | Threshold | Action Required |
|-------|-----------|-----------------|
| **❌ Error Rate** | > 5% of requests | Check logs, fix bugs |
| **🐌 Slow Response** | > 10 seconds | Optimize Pinecone/LLM calls |
| **💰 High Cost** | > $1 per day | Check for spam, add rate limits |
| **📉 Uptime** | < 99% | Check Render status page |

---

## 🐛 Common Problems & Solutions

### Problem 1: "Bot is not responding"