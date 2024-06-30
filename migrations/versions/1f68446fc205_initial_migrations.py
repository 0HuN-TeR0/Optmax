"""Initial Migrations

Revision ID: 1f68446fc205
Revises: 
Create Date: 2024-06-29 22:46:04.025410

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1f68446fc205'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('blog',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=150), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('author', sa.String(length=150), nullable=False),
    sa.Column('date', sa.DateTime(), nullable=False),
    sa.Column('category', sa.String(length=100), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('gpu_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('manufacturer', sa.Text(), nullable=True),
    sa.Column('productname', sa.Text(), nullable=True),
    sa.Column('releaseyear', sa.Float(), nullable=True),
    sa.Column('memsize', sa.Float(), nullable=True),
    sa.Column('membuswidth', sa.Float(), nullable=True),
    sa.Column('gpuclock', sa.Float(), nullable=True),
    sa.Column('memclock', sa.Float(), nullable=True),
    sa.Column('unifiedshader', sa.Float(), nullable=True),
    sa.Column('tmu', sa.Float(), nullable=True),
    sa.Column('rop', sa.Float(), nullable=True),
    sa.Column('pixelshader', sa.Float(), nullable=True),
    sa.Column('vertexshader', sa.Float(), nullable=True),
    sa.Column('igp', sa.Boolean(), nullable=True),
    sa.Column('bus', sa.Text(), nullable=True),
    sa.Column('memtype', sa.Text(), nullable=True),
    sa.Column('gpuchip', sa.Text(), nullable=True),
    sa.Column('g3dmark', sa.Float(), nullable=True),
    sa.Column('g2dmark', sa.Float(), nullable=True),
    sa.Column('price', sa.Float(), nullable=True),
    sa.Column('gpuvalue', sa.Float(), nullable=True),
    sa.Column('tdp', sa.Float(), nullable=True),
    sa.Column('powerperformance', sa.Float(), nullable=True),
    sa.Column('testdate', sa.Float(), nullable=True),
    sa.Column('category', sa.Text(), nullable=True),
    sa.Column('picture', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=150), nullable=False),
    sa.Column('email', sa.String(length=150), nullable=False),
    sa.Column('password', sa.String(length=150), nullable=False),
    sa.Column('role', sa.String(length=20), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_table('preferences',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('price_range', sa.String(length=50), nullable=True),
    sa.Column('mem_size', sa.Integer(), nullable=True),
    sa.Column('gpu_clock_range', sa.String(length=50), nullable=True),
    sa.Column('mem_clock_range', sa.String(length=50), nullable=True),
    sa.Column('unified_shader_range', sa.String(length=50), nullable=True),
    sa.Column('release_year', sa.Integer(), nullable=True),
    sa.Column('mem_type', sa.String(length=50), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('preferences')
    op.drop_table('users')
    op.drop_table('gpu_data')
    op.drop_table('blog')
    # ### end Alembic commands ###
