# Acme API Documentation v3.2

## Authentication
All API requests require an API key passed via the `Authorization` header:
```
Authorization: Bearer YOUR_API_KEY
```
API keys can be generated from the Acme Developer Portal at https://portal.acme.com/api-keys.

## Endpoints

### GET /api/v3/users
Retrieves a paginated list of users.
- **Parameters:**
  - `page` (integer, default: 1) - Page number
  - `limit` (integer, default: 20, max: 100) - Results per page
  - `status` (string) - Filter by status: "active", "inactive", "suspended"
- **Response:** JSON array of user objects with fields: id, name, email, department, status, created_at

### POST /api/v3/users
Creates a new user account.
- **Request body:**
  - `name` (string, required) - Full name
  - `email` (string, required) - Email address (must be unique)
  - `department` (string, required) - One of: "engineering", "marketing", "sales", "hr", "finance"
  - `role` (string, default: "member") - One of: "admin", "member", "viewer"
- **Response:** 201 Created with user object

### DELETE /api/v3/users/{user_id}
Deletes a user account. Requires admin role.
- **Response:** 204 No Content

### GET /api/v3/analytics
Returns usage analytics for the past 30 days.
- **Parameters:**
  - `metric` (string) - One of: "active_users", "api_calls", "error_rate"
- **Response:** JSON object with time-series data

## Rate Limits
- Free tier: 100 requests per minute
- Pro tier: 1,000 requests per minute
- Enterprise: 10,000 requests per minute

Rate limit headers are included in every response:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

## Error Codes
| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource does not exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |