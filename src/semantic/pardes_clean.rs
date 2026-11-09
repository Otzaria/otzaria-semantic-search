//! Exact pardes-clean-1: HTML, one entity pass, Hebrew quotes, explicit whitespace.
//! Keep final letters, vowels and all other characters; the tokenizer handles them.

fn space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n'
            | '\u{000b}'
            | '\u{000c}'
            | '\r'
            | ' '
            | '\u{0085}'
            | '\u{00a0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

fn entity(s: &str) -> Option<char> {
    match s {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some(' '),
        _ => {
            let (digits, radix, max) =
                if let Some(t) = s.strip_prefix("#x").or_else(|| s.strip_prefix("#X")) {
                    (t, 16, 6)
                } else {
                    (s.strip_prefix('#')?, 10, 7)
                };
            if digits.is_empty()
                || digits.len() > max
                || !digits
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && (radix == 16 || c.is_ascii_digit()))
            {
                return None;
            }
            let n = u32::from_str_radix(digits, radix).ok()?;
            if n == 0 {
                None
            } else {
                char::from_u32(n)
            }
        }
    }
}

pub(super) fn clean(text: &str) -> String {
    let mut tags = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('<') {
        tags.push_str(&rest[..i]);
        if let Some(j) = rest[i..].find('>') {
            tags.push(' ');
            rest = &rest[i + j + 1..];
        } else {
            tags.push_str(&rest[i..]);
            rest = "";
            break;
        }
    }
    tags.push_str(rest);
    let mut decoded = String::with_capacity(tags.len());
    let mut rest = tags.as_str();
    while let Some(i) = rest.find('&') {
        decoded.push_str(&rest[..i]);
        let tail = &rest[i + 1..];
        // Every recognized entity has at most eight ASCII characters before ';'.
        let candidate = tail.find(';').filter(|&j| j <= 8);
        if let Some((j, c)) = candidate.and_then(|j| entity(&tail[..j]).map(|c| (j, c))) {
            decoded.push(c);
            rest = &tail[j + 1..];
        } else {
            decoded.push('&');
            rest = tail;
        }
    }
    decoded.push_str(rest);
    let mut out = String::with_capacity(decoded.len());
    let mut pending_space = false;
    for c in decoded.chars() {
        if space(c) {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        out.push(match c {
            '״' => '"',
            '׳' => '\'',
            _ => c,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::clean;
    #[test]
    fn matches_python_reference_cases() {
        let cases: Vec<(String, String)> =
            serde_json::from_str(include_str!("../../tests/data/pardes_clean_cases.json")).unwrap();
        for (input, expected) in cases {
            assert_eq!(clean(&input), expected, "{input:?}");
        }
    }
}
