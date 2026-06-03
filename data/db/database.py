"""
SQLAlchemy async database setup for StockSense.
All ORM classes mirror the tables in data/db/schema.sql.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import (
    Column, Integer, String, Numeric, Boolean, Text,
    ARRAY, DateTime, BigInteger, ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ

from config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stocks"
    ticker = Column(String(20), primary_key=True)
    name = Column(String(200), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    exchange = Column(String(10), default="NSE")
    active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMPTZ)


class Portfolio(Base):
    __tablename__ = "portfolio"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), ForeignKey("stocks.ticker"), nullable=False)
    quantity = Column(Integer, nullable=False)
    avg_price = Column(Numeric(12, 2), nullable=False)
    buy_date = Column(TIMESTAMPTZ, nullable=False)
    active = Column(Boolean, default=True)
    notes = Column(Text)
    created_at = Column(TIMESTAMPTZ)
    updated_at = Column(TIMESTAMPTZ)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), ForeignKey("stocks.ticker"), nullable=False)
    signal_type = Column(String(10), nullable=False)
    timeframe = Column(String(20), nullable=False)
    price_at_signal = Column(Numeric(12, 2), nullable=False)
    target_price = Column(Numeric(12, 2))
    stop_loss = Column(Numeric(12, 2))
    ml_confidence = Column(Numeric(5, 4))
    kronos_confidence = Column(Numeric(5, 4))
    slm_confidence = Column(Numeric(5, 4))
    claude_confidence = Column(Numeric(5, 4))
    final_confidence = Column(Numeric(5, 4))
    status = Column(String(20), default="active")
    fired_at = Column(TIMESTAMPTZ)
    resolved_at = Column(TIMESTAMPTZ)
    actual_close = Column(Numeric(12, 2))


class SignalReasoning(Base):
    __tablename__ = "signal_reasoning"
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    reasoning = Column(Text, nullable=False)
    raw_output = Column(JSONB)
    created_at = Column(TIMESTAMPTZ)


class Learning(Base):
    __tablename__ = "learnings"
    id = Column(Integer, primary_key=True)
    learning_date = Column(DateTime, nullable=False)
    learning_type = Column(String(50), nullable=False)
    ticker = Column(String(20), ForeignKey("stocks.ticker"))
    signal_id = Column(Integer, ForeignKey("signals.id"))
    title = Column(String(300), nullable=False)
    body = Column(Text, nullable=False)
    tags = Column(ARRAY(Text))
    raw_claude_output = Column(Text)
    created_at = Column(TIMESTAMPTZ)


class ModelAccuracy(Base):
    __tablename__ = "model_accuracy"
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)
    timeframe = Column(String(20), nullable=False)
    period_start = Column(TIMESTAMPTZ, nullable=False)
    period_end = Column(TIMESTAMPTZ, nullable=False)
    total_signals = Column(Integer, default=0)
    correct_signals = Column(Integer, default=0)
    accuracy = Column(Numeric(5, 4))
    avg_confidence = Column(Numeric(5, 4))
    created_at = Column(TIMESTAMPTZ)


class FeatureImportance(Base):
    __tablename__ = "feature_importance"
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50), nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Numeric(10, 6), nullable=False)
    recorded_at = Column(TIMESTAMPTZ)


class FODaily(Base):
    """F&O aggregated per stock per day."""
    __tablename__ = "fo_daily"
    time = Column(TIMESTAMPTZ, primary_key=True, nullable=False)
    ticker = Column(String(20), ForeignKey("stocks.ticker"), primary_key=True, nullable=False)
    total_oi = Column(BigInteger)
    oi_change = Column(BigInteger)
    put_oi = Column(BigInteger)
    call_oi = Column(BigInteger)
    pcr = Column(Numeric(10, 4))


class SLMTrainingPair(Base):
    """Claude response + situation context for SLM fine-tuning."""
    __tablename__ = "slm_training_pairs"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), ForeignKey("stocks.ticker"))
    situation_date = Column(DateTime, nullable=False)
    situation_context = Column(Text, nullable=False)
    claude_response = Column(Text, nullable=False)
    created_at = Column(TIMESTAMPTZ)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
