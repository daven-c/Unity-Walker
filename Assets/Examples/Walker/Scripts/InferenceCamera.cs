using UnityEngine;
using Unity.MLAgentsExamples;

public class InferenceCamera : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 20f;
    public float fastMoveMultiplier = 3f;
    public float scrollSpeed = 10f;

    [Header("Rotation")]
    public float lookSpeed = 3f;

    float m_Yaw;
    float m_Pitch;
    CameraFollow m_CameraFollow;

    void Start()
    {
        m_CameraFollow = GetComponent<CameraFollow>();
        var angles = transform.eulerAngles;
        m_Yaw = angles.y;
        m_Pitch = angles.x;
    }

    void Update()
    {
        // Toggle free camera with Tab
        if (Input.GetKeyDown(KeyCode.Tab))
        {
            if (m_CameraFollow != null)
                m_CameraFollow.enabled = !m_CameraFollow.enabled;
        }

        // Only control camera when CameraFollow is disabled
        if (m_CameraFollow != null && m_CameraFollow.enabled)
            return;

        // Right-click drag to rotate
        if (Input.GetMouseButton(1))
        {
            m_Yaw += Input.GetAxis("Mouse X") * lookSpeed;
            m_Pitch -= Input.GetAxis("Mouse Y") * lookSpeed;
            m_Pitch = Mathf.Clamp(m_Pitch, -89f, 89f);
            transform.rotation = Quaternion.Euler(m_Pitch, m_Yaw, 0f);
        }

        // WASD + QE movement
        float speed = moveSpeed * (Input.GetKey(KeyCode.LeftShift) ? fastMoveMultiplier : 1f) * Time.unscaledDeltaTime;
        if (Input.GetKey(KeyCode.W)) transform.position += transform.forward * speed;
        if (Input.GetKey(KeyCode.S)) transform.position -= transform.forward * speed;
        if (Input.GetKey(KeyCode.A)) transform.position -= transform.right * speed;
        if (Input.GetKey(KeyCode.D)) transform.position += transform.right * speed;
        if (Input.GetKey(KeyCode.Q)) transform.position -= transform.up * speed;
        if (Input.GetKey(KeyCode.E)) transform.position += transform.up * speed;

        // Scroll to zoom forward/back
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (scroll != 0f)
            transform.position += transform.forward * scroll * scrollSpeed;
    }
}
