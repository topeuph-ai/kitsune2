//! Protocol types for the sharding module.

include!("../../proto/gen/kitsune2.gossip.sharding.rs");

use kitsune2_api::{K2Error, K2Result};
use prost::Message;

/// Deserialize a sharding message.
pub(crate) fn deserialize_sharding_message(
    value: bytes::Bytes,
) -> K2Result<K2ShardingMessage> {
    K2ShardingMessage::decode(value).map_err(|e| {
        K2Error::other_src("Failed to deserialize sharding message", e)
    })
}

/// Serialize a sharding message.
pub(crate) fn serialize_sharding_message(
    value: K2ShardingMessage,
) -> bytes::Bytes {
    let mut out = bytes::BytesMut::new();

    // Encoding can only fail if the buffer is too small, and `BytesMut`
    // grows as needed, so this cannot fail.
    value.encode(&mut out).expect("Failed to encode message");

    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrink_intent_round_trip() {
        let msg = K2ShardingMessage {
            msg: Some(k2_sharding_message::Msg::ShrinkIntent(
                K2ShardingShrinkIntentMessage {
                    agent_id: bytes::Bytes::from_static(b"test-agent"),
                    vacate_start: 7 << 23,
                    vacate_end: (15 << 23) - 1,
                    expires_at_us: 1_234_567,
                },
            )),
        };

        let decoded = deserialize_sharding_message(serialize_sharding_message(
            msg.clone(),
        ))
        .unwrap();

        assert_eq!(msg, decoded);
    }
}
