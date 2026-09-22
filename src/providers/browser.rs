use std::process::{Command, Stdio};

pub(crate) fn open(url: &str, headless: bool) {
    open_with(
        url,
        headless,
        std::env::consts::OS,
        graphical_session(),
        |program, url| {
            let mut child = Command::new(program)
                .arg(url)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()?;
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            Ok(())
        },
    );
}

fn graphical_session() -> bool {
    ["DISPLAY", "WAYLAND_DISPLAY"]
        .into_iter()
        .filter_map(std::env::var_os)
        .any(|value| !value.is_empty())
}

fn launcher(os: &str, headless: bool, graphical: bool) -> Option<&'static str> {
    if headless {
        return None;
    }
    match os {
        "macos" => Some("open"),
        "linux" if graphical => Some("xdg-open"),
        _ => None,
    }
}

fn open_with<F>(url: &str, headless: bool, os: &str, graphical: bool, launch: F)
where
    F: FnOnce(&str, &str) -> std::io::Result<()>,
{
    if let Some(program) = launcher(os, headless, graphical) {
        let _ = launch(program, url);
    }
}

#[cfg(test)]
mod tests {
    use super::open_with;
    use std::{cell::Cell, io};

    #[test]
    fn skips_headless_browserless_linux_and_unsupported_systems() {
        let calls = Cell::new(0);
        for (os, headless, graphical) in [
            ("macos", true, true),
            ("linux", false, false),
            ("windows", false, true),
        ] {
            open_with("https://example.com", headless, os, graphical, |_, _| {
                calls.set(calls.get() + 1);
                Ok(())
            });
        }
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn passes_the_complete_url_as_one_argument() {
        let invocation = std::cell::RefCell::new(None);
        let url = "https://example.com/login?code=a%2Bb&state=secret";
        open_with(url, false, "linux", true, |program, argument| {
            invocation.replace(Some((program.to_owned(), argument.to_owned())));
            Ok(())
        });
        assert_eq!(
            invocation.into_inner(),
            Some(("xdg-open".to_owned(), url.to_owned()))
        );
    }

    #[test]
    fn launcher_failure_is_nonfatal() {
        let calls = Cell::new(0);
        open_with("https://example.com", false, "macos", false, |_, _| {
            calls.set(calls.get() + 1);
            Err(io::Error::new(io::ErrorKind::NotFound, "missing opener"))
        });
        assert_eq!(calls.get(), 1);
    }
}
