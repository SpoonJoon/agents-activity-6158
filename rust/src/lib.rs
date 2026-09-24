use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

fn is_number(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s == "0" {
        return true;
    }
    if s.starts_with('0') {
        return false;
    }
    s.bytes().all(|b| b.is_ascii_digit())
}

fn is_prerelease_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    // Identifiers consisting of digits SHOULD NOT contain leading zeroes (SemVer item 9)
    if s.bytes().all(|b| b.is_ascii_digit()) {
        return is_number(s);
    }
    s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

fn is_build_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

pub fn parse(s: &str) -> Result<Version, String> {
    if s.is_empty() {
        return Err("empty string".into());
    }

    // Split off build metadata (first '+' separates version/prerelease from build)
    let (core_pre, build) = match s.find('+') {
        Some(idx) => {
            let cp = &s[..idx];
            let b = &s[idx + 1..];
            if b.is_empty() {
                return Err("empty build".into());
            }
            (cp, Some(b))
        }
        None => (s, None),
    };

    // Validate build identifiers if present
    if let Some(b) = build {
        for part in b.split('.') {
            if !is_build_identifier(part) {
                return Err("invalid build identifier".into());
            }
        }
    }

    // Split off prerelease (first '-' separates core version from prerelease)
    let (core, prerelease) = match core_pre.find('-') {
        Some(idx) => {
            let c = &core_pre[..idx];
            let p = &core_pre[idx + 1..];
            if p.is_empty() {
                return Err("empty prerelease".into());
            }
            (c, Some(p))
        }
        None => (core_pre, None),
    };

    // Validate prerelease identifiers if present
    if let Some(p) = prerelease {
        for part in p.split('.') {
            if !is_prerelease_identifier(part) {
                return Err("invalid prerelease identifier".into());
            }
        }
    }

    // Parse core version (major.minor.patch)
    let nums: Vec<&str> = core.split('.').collect();
    if nums.len() != 3 {
        return Err("invalid core version format".into());
    }

    let major_str = nums[0];
    let minor_str = nums[1];
    let patch_str = nums[2];

    if !is_number(major_str) {
        return Err("invalid major version".into());
    }
    if !is_number(minor_str) {
        return Err("invalid minor version".into());
    }
    if !is_number(patch_str) {
        return Err("invalid patch version".into());
    }

    let major = major_str.parse().map_err(|_| "invalid major")?;
    let minor = minor_str.parse().map_err(|_| "invalid minor")?;
    let patch = patch_str.parse().map_err(|_| "invalid patch")?;

    Ok(Version {
        major,
        minor,
        patch,
        prerelease: prerelease.map(String::from),
        build: build.map(String::from),
    })
}

pub fn to_string(v: &Version) -> String {
    let mut s = format!("{}.{}.{}", v.major, v.minor, v.patch);
    if let Some(ref pre) = v.prerelease {
        s.push('-');
        s.push_str(pre);
    }
    if let Some(ref bld) = v.build {
        s.push('+');
        s.push_str(bld);
    }
    s
}

pub fn compare(a: &Version, b: &Version) -> Ordering {
    match a.major.cmp(&b.major) {
        Ordering::Equal => {}
        ord => return ord,
    }
    match a.minor.cmp(&b.minor) {
        Ordering::Equal => {}
        ord => return ord,
    }
    match a.patch.cmp(&b.patch) {
        Ordering::Equal => {}
        ord => return ord,
    }

    match (&a.prerelease, &b.prerelease) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(ra), Some(rb)) => nat_cmp(ra, rb),
    }
}

fn nat_cmp(a: &str, b: &str) -> Ordering {
    let a_parts: Vec<&str> = a.split('.').collect();
    let b_parts: Vec<&str> = b.split('.').collect();
    for (sub_a, sub_b) in a_parts.iter().zip(b_parts.iter()) {
        let a_int = sub_a.parse::<u64>();
        let b_int = sub_b.parse::<u64>();
        let ord = match (a_int, b_int) {
            (Ok(ai), Ok(bi)) => ai.cmp(&bi),
            (Ok(_), Err(_)) => Ordering::Less,
            (Err(_), Ok(_)) => Ordering::Greater,
            (Err(_), Err(_)) => sub_a.cmp(sub_b),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a_parts.len().cmp(&b_parts.len())
}

pub fn bump_major(v: &Version) -> Version {
    Version {
        major: v.major + 1,
        minor: 0,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

pub fn bump_minor(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor + 1,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

pub fn bump_patch(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor,
        patch: v.patch + 1,
        prerelease: None,
        build: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsing() {
        assert!(parse("1.2.3").is_ok());
        assert!(parse("01.2.3").is_err());
        assert!(parse("1.0.0-").is_err());
        assert!(parse("1.0.0+").is_err());
    }

    #[test]
    fn test_compare() {
        let v1 = parse("1.0.0").unwrap();
        let v2 = parse("2.0.0").unwrap();
        assert_eq!(compare(&v1, &v2), Ordering::Less);
    }

    #[test]
    fn test_bump() {
        let v = parse("1.2.3").unwrap();
        assert_eq!(to_string(&bump_major(&v)), "2.0.0");
        assert_eq!(to_string(&bump_minor(&v)), "1.3.0");
        assert_eq!(to_string(&bump_patch(&v)), "1.2.4");
    }
}
