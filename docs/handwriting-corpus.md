# Create an initial personal handwriting corpus

This guide describes how to make the initial handwriting corpus for a future
Sibyl handwriting-adaptation experiment. It is an initial corpus of about 20
pages, not a claim that 20 pages will be sufficient for final fine-tuning.

The corpus teaches:

```text
visual handwriting → exact transcription
```

It is not intended to teach:

```text
topic → likely word
```

Write generic English and natural personal notes. The useful signal is your
handwriting, not specialized vocabulary or subject matter.

## Before you write

### Write naturally

Use your actual handwriting. Do not make it unusually neat, slow down to make
letters beautiful, deliberately make it messy, or invent a special “training
handwriting” style. The goal is to capture your natural handwriting
distribution.

Use the pen you normally use for notes, and keep the writing instrument
consistent throughout this controlled corpus. Use plain white paper. Use
unruled paper for the character and word calibration pages, and ordinary
notebook paper for natural-note pages. No special paper is required.

Repeated examples should be written naturally rather than copied as identical
glyphs. For example, four occurrences of `the` should be four ordinary
occurrences, not four attempts to reproduce the same shapes. That variation is
valuable because the model needs to learn your range of letter forms.

Leave ordinary handwriting variation alone. If you make an actual writing
mistake, you may cross it out and rewrite it naturally. The transcription is
the intended final text; crossed-out text is not part of that transcription
unless it is explicitly collected later as a separate research target. Do not
manufacture errors.

### Capture the pages

When scanning or photographing pages:

- Keep the camera or scanner parallel to the page.
- Use even lighting and avoid shadows.
- Capture the entire page.
- Use the highest practical source resolution.
- Preserve the original files.
- Prefer lossless PNG where practical.

Do not crop individual words during collection. Do not resize the original
files. Do not run OCR on the originals before establishing their ground truth.
The page image and its exact transcription are the canonical collected datum.

## Page IDs and ground truth

Use stable page IDs:

```text
HW001
HW002
...
HW020
```

The eventual files should correspond as pairs:

```text
HW001.png
HW001.txt

HW002.png
HW002.txt

...
```

Each `.txt` file contains the exact intended final text written on that page.
For controlled word pages, write the words in their written order. For
sentence and natural-note pages, use the exact supplied sentences and
paragraphs. Do not include page IDs, instructions, commentary, model
predictions, or OCR-system corrections in the `.txt` files.

Preserve the exact intended capitalization and punctuation, including
apostrophes, commas, periods, question marks, quotation marks, hyphens, and
numbers.

The distinction is important:

```text
ground truth = what you intended and wrote
model output  = an observation that may be wrong
```

The model's output must never be fed back into the ground-truth corpus without
human verification. Semantic context can make an incorrect handwriting
reading look plausible.

## The 20-page progression

Follow this progression:

```text
characters
    ↓
character combinations
    ↓
common English words
    ↓
short sentences
    ↓
natural notes
    ↓
deliberately useful difficult combinations
```

Write each page as a page, keeping its original layout in the image. Do not
make individual word crops.

### HW001 — lowercase alphabet

Write each lowercase character approximately ten times:

```text
a a a a a a a a a a
b b b b b b b b b b
...
z z z z z z z z z z
```

### HW002 — uppercase alphabet

Write each uppercase character approximately eight times:

```text
A A A A A A A A
B B B B B B B B
...
Z Z Z Z Z Z Z Z
```

### HW003 — numbers and visual confusions

Write each digit ten times:

```text
0 0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 1
2 2 2 2 2 2 2 2 2 2
3 3 3 3 3 3 3 3 3 3
4 4 4 4 4 4 4 4 4 4
5 5 5 5 5 5 5 5 5 5
6 6 6 6 6 6 6 6 6 6
7 7 7 7 7 7 7 7 7 7
8 8 8 8 8 8 8 8 8 8
9 9 9 9 9 9 9 9 9 9
```

Then write these visual-confusion sequences eight times as shown:

```text
O 0 O 0 O 0 O 0
I l 1 I l 1 I l 1
S 5 S 5 S 5 S 5
B 8 B 8 B 8 B 8
G 6 G 6 G 6 G 6
Z 2 Z 2 Z 2 Z 2
```

### HW004–HW010 — common English words

Write each word on its page in the listed order.

#### HW004 — each word four times

```text
the
of
and
to
in
a
is
that
for
it
as
was
with
be
on
by
this
are
from
or
at
not
but
what
all
```

#### HW005 — each word four times

```text
were
when
we
there
can
an
your
which
their
said
if
do
will
each
about
how
up
out
them
then
she
many
some
so
these
```

#### HW006 — each word four times

```text
would
other
into
has
more
her
two
like
him
see
time
could
no
make
than
first
been
its
who
now
people
may
way
use
over
```

#### HW007 — each word four times

```text
only
new
also
very
after
most
because
where
those
under
through
back
good
before
here
work
right
think
well
much
still
even
these
never
another
```

#### HW008 — each word three times

```text
one
two
three
four
five
six
seven
eight
nine
ten
first
second
next
last
same
different
another
again
before
after
then
now
here
there
where
```

#### HW009 — each word three times

```text
this
that
these
those
there
their
they
them
then
than
when
where
what
which
while
with
without
through
though
thought
three
right
write
world
would
```

#### HW010 — each word three times

```text
people
little
middle
simple
something
nothing
another
together
between
different
important
possible
problem
really
around
always
already
almost
enough
everything
something
sometimes
without
within
whether
```

### HW011–HW015 — sentence pages

Write these sentences exactly, including punctuation.

#### HW011

```text
The time is now.
This is the way.
That was the first one.
I went to the store.
We went home after work.
The weather was good.
There is more to do.
I think this will work.
That is what I meant.
We can do it later.
```

#### HW012

```text
I was there when you called.
We can talk about it later.
The first thing is to make a plan.
I do not know what they want.
This is different from the other one.
There are many ways to do this.
I thought it would be easier.
She said that everything was fine.
We went back to the same place.
I will see you tomorrow.
```

#### HW013

```text
The work is almost finished.
There is still a little more to do.
I want to make sure everything is ready.
This should be enough for now.
We can come back to it later.
The important thing is to keep moving.
I do not think we need another one.
That is probably the best way to do it.
Sometimes the simple answer is the right one.
It is better to take the time to do it well.
```

#### HW014

```text
What is the time?
Where are you going?
Is this the right place?
I think so.
Maybe we should wait.
No, that is not what I meant.
Yes, we can do that.
It was cold, but the water was calm.
I said, "We can do it later."
The answer is simple: keep going.
```

#### HW015

```text
I have two things to do.
There are three people here.
The first one was better.
I waited for ten minutes.
It took about twenty minutes.
The number was 42.
I wrote down 17 and 28.
The year was 2026.
There were 100 people there.
I only need one more.
```

### HW016–HW018 — natural writing

Write these pages as naturally as possible while preserving the supplied
ground truth. Keep the paragraph breaks shown here.

#### HW016

```text
The first thing I need to do is finish this before the end of the day.

There are a few things that still need to be checked before we move on.

I think the simplest way is to start with what we already know.

The important part is making sure the whole thing works together.

I can come back to the details after the main work is finished.

There is no reason to make this more complicated than it needs to be.
```

#### HW017

```text
Need to remember this for later.

Call back tomorrow morning.

Check the list before leaving.

Pick up the things we need on the way home.

Finish this first, then move on to the next thing.

Look at the other option before making a decision.

Write this down so I do not forget it.

Come back to this when there is more time.
```

#### HW018

```text
I went out early this morning and walked for a while.

The weather was warm, but there was a good breeze.

I stopped for coffee and spent some time thinking about what to do next.

There are still a few things I want to finish before tonight.

After that I can take a break and come back to the rest tomorrow.
```

### HW019 — visually difficult common words

Write each word three times. Do not alter the words to make them difficult; the
difficulty should come from your natural handwriting.

```text
minimum
maximum
little
middle
million
common
coming
running
writing
written
really
early
every
never
level
letter
better
matter
between
different
```

### HW020 — final natural page

Write this as ordinary natural notes rather than as a worksheet:

```text
Today I want to finish the things that have been sitting around for too long.

Some of them are easy and some will take more time.

The important thing is to keep moving and not get stuck trying to make everything perfect.

I can finish the simple things first and come back to the harder ones later.

There is always another problem to solve, but there is also usually a simple way to begin.

Once the first part is done, the rest tends to become much easier.
```

## Use several writing sessions

Complete the corpus across several sessions rather than in one marathon:

```text
Session 1: HW001–HW003
Session 2: HW004–HW007
Session 3: HW008–HW010
Session 4: HW011–HW015
Session 5: HW016–HW020
```

Natural handwriting varies with fatigue, speed, day, and context. That
variation is useful evidence of the handwriting the future adapter must handle.

## What happens later

This corpus is being collected for future handwriting adaptation. The eventual
conceptual pipeline is:

```text
handwritten source pages
        +
exact ground truth
        ↓
training dataset
        ↓
Qwen3-VL parameter-efficient adaptation
        ↓
personal handwriting adapter
        ↓
evaluation on unseen handwriting
```

Future work will determine the exact Qwen3-VL training format, the
train/validation/test split, the LoRA or QLoRA configuration, training
hardware, and evaluation procedure. Those decisions are not established by
this collection guide. No training command is provided here.
