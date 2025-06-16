﻿using System.Collections.Generic;
using Microsoft.AspNetCore.Mvc;
using System;
using Quantia.Models;

namespace Quantia.Controllers
{
    public class TechnicalAnalysisController : Controller
    {
        public ActionResult Index()        
        {
            var model = new TechnicalAnalysisViewModel
            {
                StartDate = new DateTime(2023, 7, 31),
                EndDate = new DateTime(2023, 7, 31),
                AnalysisDate = new DateTime(2023, 7, 31),
                SignalsDetected = new List<string>
                {
                    "RSI crossover above 50",
                    "MACD bullish crossover",
                    "20-Day MA rising",
                    "Bullish engulfing"
                },
                Entry = 1985,
                StopLoss = 1945,
                TakeProfit = 2040,
                RiskReward = "50:140"
            };

            return View(model);
        }
    }
}
