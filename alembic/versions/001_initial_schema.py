"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create tasks table
    op.create_table(
        'tasks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('agent_type', sa.String(50), nullable=False, index=True),
        sa.Column('instructions', sa.Text(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False, default=dict),
        sa.Column('context_data', sa.JSON(), nullable=False, default=dict),
        sa.Column('status', sa.String(20), nullable=False, default='pending', index=True),
        sa.Column('progress', sa.String(255), nullable=True),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('artifacts', sa.JSON(), nullable=False, default=list),
        sa.Column('parent_task_id', sa.String(36), sa.ForeignKey('tasks.id'), nullable=True),
        sa.Column('llm_calls', sa.Integer(), nullable=False, default=0),
        sa.Column('total_tokens', sa.Integer(), nullable=False, default=0),
        sa.Column('cost_cents', sa.Float(), nullable=False, default=0.0),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    
    # Create task_messages table
    op.create_table(
        'task_messages',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('task_id', sa.String(36), sa.ForeignKey('tasks.id'), nullable=False, index=True),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False, default=''),
        sa.Column('tool_calls', sa.JSON(), nullable=True),
        sa.Column('tool_results', sa.JSON(), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('task_messages')
    op.drop_table('tasks')
