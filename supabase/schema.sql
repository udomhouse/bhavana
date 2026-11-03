-- Bhāvanā · สนทนาธรรม (community Q&A)
-- 1. Supabase › SQL Editor › paste this file › Run
-- 2. Authentication › Sign In / Providers › turn on "Allow anonymous sign-ins"
-- Each phone signs in anonymously (no email), so a person is one device's account.

create table public.banned_users (
  user_id uuid primary key references auth.users(id) on delete cascade,
  reason text,
  created_at timestamptz not null default now()
);

create table public.questions (
  id bigint generated always as identity primary key,
  author uuid not null default auth.uid() references auth.users(id) on delete cascade,
  alias text not null check (char_length(alias) between 1 and 40),
  title text not null check (char_length(title) between 4 and 140),
  body text not null default '' check (char_length(body) <= 2000),
  topic text not null default 'ทั่วไป' check (topic in ('ครอบครัว','การงาน','ความรัก','สุขภาพใจ','การภาวนา','ทั่วไป')),
  answer_count int not null default 0,
  sadhu_count int not null default 0,
  report_count int not null default 0,
  hidden boolean not null default false,
  created_at timestamptz not null default now()
);

create table public.answers (
  id bigint generated always as identity primary key,
  question_id bigint not null references public.questions(id) on delete cascade,
  author uuid not null default auth.uid() references auth.users(id) on delete cascade,
  alias text not null check (char_length(alias) between 1 and 40),
  body text not null check (char_length(body) between 1 and 2000),
  sadhu_count int not null default 0,
  report_count int not null default 0,
  hidden boolean not null default false,
  created_at timestamptz not null default now()
);
create index answers_by_question on public.answers (question_id, created_at);

create table public.reactions (            -- "สาธุ"
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  target_type text not null check (target_type in ('q','a')),
  target_id bigint not null,
  created_at timestamptz not null default now(),
  primary key (user_id, target_type, target_id)
);

create table public.reports (
  id bigint generated always as identity primary key,
  reporter uuid not null default auth.uid() references auth.users(id) on delete cascade,
  target_type text not null check (target_type in ('q','a')),
  target_id bigint not null,
  reason text not null check (char_length(reason) <= 200),
  created_at timestamptz not null default now(),
  unique (reporter, target_type, target_id)
);

alter table public.banned_users enable row level security;
alter table public.questions   enable row level security;
alter table public.answers     enable row level security;
alter table public.reactions   enable row level security;
alter table public.reports     enable row level security;

create policy "read visible questions" on public.questions for select using (not hidden);
create policy "read visible answers"   on public.answers   for select using (not hidden);
create policy "read own reactions"     on public.reactions for select using (user_id = auth.uid());

create policy "ask" on public.questions for insert
  with check (author = auth.uid() and not exists (select 1 from public.banned_users b where b.user_id = auth.uid()));
create policy "answer" on public.answers for insert
  with check (author = auth.uid() and not exists (select 1 from public.banned_users b where b.user_id = auth.uid()));
create policy "delete own question" on public.questions for delete using (author = auth.uid());
create policy "delete own answer"   on public.answers   for delete using (author = auth.uid());
create policy "give sadhu"   on public.reactions for insert with check (user_id = auth.uid());
create policy "remove sadhu" on public.reactions for delete using (user_id = auth.uid());
create policy "report" on public.reports for insert with check (reporter = auth.uid());

-- Counters are kept by triggers (security definer, so they can update rows the user doesn't own).
create function public.bump_answer_count() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  if tg_op = 'INSERT' then
    update questions set answer_count = answer_count + 1 where id = new.question_id;
  else
    update questions set answer_count = greatest(answer_count - 1, 0) where id = old.question_id;
  end if;
  return null;
end $$;
create trigger answers_count after insert or delete on public.answers
  for each row execute function public.bump_answer_count();

create function public.bump_sadhu() returns trigger
language plpgsql security definer set search_path = public as $$
declare
  r reactions;
  d int;
begin
  if tg_op = 'INSERT' then r := new; d := 1; else r := old; d := -1; end if;
  if r.target_type = 'q' then
    update questions set sadhu_count = greatest(sadhu_count + d, 0) where id = r.target_id;
  else
    update answers set sadhu_count = greatest(sadhu_count + d, 0) where id = r.target_id;
  end if;
  return null;
end $$;
create trigger reactions_count after insert or delete on public.reactions
  for each row execute function public.bump_sadhu();

-- 3 reports hide a post automatically; un-hide or delete it from the Table Editor.
create function public.on_report() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  if new.target_type = 'q' then
    update questions set report_count = report_count + 1, hidden = hidden or report_count + 1 >= 3 where id = new.target_id;
  else
    update answers set report_count = report_count + 1, hidden = hidden or report_count + 1 >= 3 where id = new.target_id;
  end if;
  return null;
end $$;
create trigger reports_hide after insert on public.reports
  for each row execute function public.on_report();
