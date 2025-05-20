use crate::utils;

#[derive(Clone)]
pub struct Session {
    pub session_id: String,
    pub last_updated_time: u64,
}

impl Session {
    pub fn new(session_id: String) -> Session {
        Session {
            session_id,
            last_updated_time: utils::now(),
        }
    }

    pub fn update(&mut self) {
        self.last_updated_time = utils::now();
    }
}

pub struct SessionManager {
    sessions: Vec<Session>,
}

impl SessionManager {
    pub fn new() -> SessionManager {
        SessionManager {
            sessions: Vec::new(),
        }
    }

    pub fn create_session(&mut self) -> &Session {
        let session_id: String = utils::generate_session_id();
        let session = Session::new(session_id.clone());
        self.sessions.push(session);

        self.sessions.last().unwrap()
    }
}
