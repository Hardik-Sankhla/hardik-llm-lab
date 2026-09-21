# Motivation

## Why I Started

I wanted to understand what actually happens inside a language model. Not just how to use an API, but how the architecture works, why each component exists, and what happens when you change things.

## The Problem with Tutorials

Most tutorials teach you to:
1. Copy code
2. Run it
3. Get a result

But they don't teach you:
- Why this specific architecture?
- What happens if I remove this component?
- How does this scale?
- What breaks and why?

## My Approach

Instead of following a tutorial, I'm building a **research laboratory**:

1. **Learn** - Study the concept from the reference material
2. **Rebuild** - Implement independently from first principles
3. **Verify** - Write tests, compare against reference
4. **Experiment** - Run controlled ablations
5. **Break** - Push to failure, document what breaks
6. **Explain** - Document findings with evidence
7. **Extend** - Go beyond the curriculum

This repository is the artifact of that process.

## What This Is Not

- ❌ A chapter-by-chapter copy of the book
- ❌ A "I built an LLM" brag repo
- ❌ A collection of unexamined notebooks

## What This Is

- ✅ A reproducible research environment
- ✅ Controlled experiments with hypotheses
- ✅ Documented failures and discoveries
- ✅ A system I understand deeply enough to modify