from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Numeric, Boolean, Text, ARRAY, DateTime, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense")

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
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


class Portfolio(Base):
    __tablename__ = "portfolio"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), ForeignKey("stocks.ticker"), nullable=False)
    quantity = Column(Integer, nullable=False)
    avg_price = Column(Numeric(12, 2), nullable=False)
    buy_date = Column(TIMESTAMPTZ, nullable=False)
    active = Column(Boolean, default=True)
    notes = Column(Text)


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
    actual_close = Column(Numeric(12, 2))


class SignalReasoning(Base):
    __tablename__ = "signal_reasoning"
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    reasoning = Column(Text, nullable=False)
    raw_output = Column(JSONB)


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


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
