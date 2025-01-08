section_patterns = {
    'pattern1': [
        ("Intro", 8),
        ("Verse1", 16),
        ("PreChorus", 8),
        ("Chorus1", 16),
        ("Verse2", 16),
        ("PreChorus", 8),
        ("Chorus2", 16),
        ("Bridge", 8),
        ("Chorus3", 16),
        ("Outro", 8)
    ],
    'pattern2': [
        ("Intro", 8),
        ("Verse1", 16),
        ("Chorus1", 8),
        ("Verse2", 16),
        ("Chorus2", 8),
        ("Bridge", 8),
        ("Chorus3", 8),
        ("Outro", 8)
    ],
    'pattern3': [
        ("Intro", 8),
        ("Verse1", 16),
        ("PreChorus1", 8),
        ("Chorus1", 16),
        ("Verse2", 16),
        ("PreChorus2", 8),
        ("Chorus2", 16),
        ("Bridge", 8),
        ("Chorus3", 16),
        ("Outro", 8)
    ],
    'pattern4': [
        ("Intro", 8),
        ("Chorus", 8),
        ("Verse", 16),
        ("PreChorus", 4),
        ("Chorus", 8),
        ("Verse", 16),
        ("PreChorus", 4),
        ("Chorus", 8),
        ("Outro", 8)
    ]
}

def form_prompt_structure(pattern: str):
    structure = section_patterns[pattern]
    return "->".join([f"{x[0]}({x[1]})" for x in structure])

def total_bars(pattern: str):
    structure = section_patterns[pattern]
    return sum(x[1] for x in structure)

def form_pattern_string_list(pattern: str):
    structure = section_patterns[pattern]
    r = []
    for x in structure:
        r.extend([x[0] for _ in range(x[1])])
    return r

def form_stat_prompt(pattern_str_list: str, from_bar: int, to_bar: int):
    belong_sections = pattern_str_list[from_bar: to_bar]
    sections_stat = []
    for s in belong_sections:
        if not sections_stat:
            sections_stat.append([s, 1])
            continue
        if sections_stat[-1][0] == s:
            sections_stat[-1][1] = sections_stat[-1][1]+1
        else:
            sections_stat.append([s, 1])
    sections_stat = [f"{count} bars of {label}" for label, count in sections_stat]
    if len(sections_stat) > 1:
        sections_stat[-1] = "and " + sections_stat[-1]
    return ", ".join(sections_stat)

if __name__ == "__main__":
    print(form_prompt_structure('pattern1'))
    print(total_bars('pattern1'))
    psl = form_pattern_string_list('pattern1')
    print(psl)
    print(form_stat_prompt(psl, 4, 25))