# CloudSync API Documentation

## Overview
CloudSync API allows you to manage files, users, and synchronization tasks programmatically. The API uses REST principles and returns JSON responses.

**Base URL:** `https://api.cloudsync.com/v1`

## Authentication

### API Key Authentication
Include your API key in the Authorization header:
```
Authorization: Bearer cs_live_1234567890abcdef
```

### Rate Limits
- **Free tier:** 100 requests/hour
- **Pro tier:** 1000 requests/hour
- **Enterprise:** 10000 requests/hour

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per hour
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## File Management

### Upload File
**POST** `/files`

Upload a new file to your CloudSync storage.

**Request Headers:**
- `Content-Type: multipart/form-data`
- `Authorization: Bearer YOUR_API_KEY`

**Request Body:**
```json
{
  "file": "[binary data]",
  "filename": "document.pdf",
  "folder_id": "folder_123",
  "description": "Important contract document"
}
```

**Response (201 Created):**
```json
{
  "id": "file_456",
  "filename": "document.pdf", 
  "size": 1024000,
  "upload_date": "2024-01-15T10:30:00Z",
  "folder_id": "folder_123",
  "download_url": "https://cdn.cloudsync.com/files/file_456",
  "expires_at": "2024-01-22T10:30:00Z"
}
```

### Get File Details
**GET** `/files/{file_id}`

Retrieve metadata for a specific file.

**Response (200 OK):**
```json
{
  "id": "file_456",
  "filename": "document.pdf",
  "size": 1024000,
  "content_type": "application/pdf",
  "upload_date": "2024-01-15T10:30:00Z",
  "last_modified": "2024-01-15T10:30:00Z",
  "folder_id": "folder_123",
  "tags": ["contract", "legal"],
  "shared": false,
  "download_count": 5
}
```

### Delete File
**DELETE** `/files/{file_id}`

Permanently delete a file. This action cannot be undone.

**Response (204 No Content)**

## User Management

### Create User
**POST** `/users`

Create a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "first_name": "John",
  "last_name": "Doe",
  "plan": "pro"
}
```

**Response (201 Created):**
```json
{
  "id": "user_789",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "plan": "pro",
  "created_at": "2024-01-15T10:30:00Z",
  "storage_used": 0,
  "storage_limit": 107374182400,
  "verified": false
}
```

### Get User Profile
**GET** `/users/me`

Get the current authenticated user's profile information.

**Response (200 OK):**
```json
{
  "id": "user_789",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "plan": "pro",
  "created_at": "2024-01-15T10:30:00Z",
  "last_login": "2024-01-20T08:15:00Z",
  "storage_used": 5368709120,
  "storage_limit": 107374182400,
  "verified": true,
  "two_factor_enabled": false
}
```

### Update User Profile
**PATCH** `/users/me`

Update the current user's profile information.

**Request Body:**
```json
{
  "first_name": "Jane",
  "last_name": "Smith",
  "two_factor_enabled": true
}
```

## Synchronization

### Start Sync Job
**POST** `/sync/jobs`

Start a new synchronization job between local and cloud storage.

**Request Body:**
```json
{
  "source_folder": "folder_123",
  "destination": "local",
  "sync_mode": "bidirectional",
  "delete_missing": false,
  "exclude_patterns": ["*.tmp", "*.log"]
}
```

**Response (201 Created):**
```json
{
  "job_id": "sync_001",
  "status": "running",
  "created_at": "2024-01-20T14:30:00Z",
  "progress": 0,
  "files_processed": 0,
  "files_total": 150,
  "estimated_completion": "2024-01-20T14:45:00Z"
}
```

### Get Sync Status
**GET** `/sync/jobs/{job_id}`

Check the status of a running synchronization job.

**Response (200 OK):**
```json
{
  "job_id": "sync_001",
  "status": "completed",
  "created_at": "2024-01-20T14:30:00Z",
  "completed_at": "2024-01-20T14:42:30Z",
  "progress": 100,
  "files_processed": 150,
  "files_total": 150,
  "files_uploaded": 45,
  "files_downloaded": 12,
  "files_updated": 8,
  "files_skipped": 85,
  "bytes_transferred": 2147483648
}
```

## Error Handling

### HTTP Status Codes
- **200 OK** - Request successful
- **201 Created** - Resource created successfully
- **204 No Content** - Request successful, no response body
- **400 Bad Request** - Invalid request parameters
- **401 Unauthorized** - Invalid or missing API key
- **403 Forbidden** - Access denied to resource
- **404 Not Found** - Resource not found
- **429 Too Many Requests** - Rate limit exceeded
- **500 Internal Server Error** - Server error

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "File type not supported. Allowed types: jpg, png, pdf, doc, docx",
    "details": {
      "received_type": "exe",
      "allowed_types": ["jpg", "png", "pdf", "doc", "docx"]
    },
    "request_id": "req_abc123"
  }
}
```

## Webhooks

### Setting Up Webhooks
**POST** `/webhooks`

Register a webhook endpoint to receive real-time notifications.

**Request Body:**
```json
{
  "url": "https://yourapp.com/webhooks/cloudsync",
  "events": ["file.uploaded", "file.deleted", "sync.completed"],
  "secret": "webhook_secret_key"
}
```

### Webhook Events
- `file.uploaded` - New file uploaded
- `file.deleted` - File deleted
- `file.shared` - File shared with another user
- `sync.started` - Sync job started
- `sync.completed` - Sync job completed
- `sync.failed` - Sync job failed
- `user.created` - New user registered
- `user.deleted` - User account deleted

### Webhook Payload Example
```json
{
  "event": "file.uploaded",
  "timestamp": "2024-01-20T15:30:00Z",
  "data": {
    "file_id": "file_456",
    "filename": "report.pdf",
    "user_id": "user_789",
    "size": 2048000
  }
}
```

## SDK and Libraries

### Official SDKs
- **JavaScript/Node.js:** `npm install cloudsync-sdk`
- **Python:** `pip install cloudsync-python`
- **PHP:** `composer require cloudsync/php-sdk`
- **Ruby:** `gem install cloudsync`

### Quick Start Example (JavaScript)
```javascript
const CloudSync = require('cloudsync-sdk');

const client = new CloudSync({
  apiKey: 'cs_live_1234567890abcdef'
});

// Upload a file
const file = await client.files.upload({
  file: fs.createReadStream('document.pdf'),
  filename: 'document.pdf',
  folder_id: 'folder_123'
});

console.log('File uploaded:', file.id);
```

## Support

### Getting Help
- **Documentation:** https://docs.cloudsync.com
- **Support Email:** support@cloudsync.com
- **Community Forum:** https://community.cloudsync.com
- **Status Page:** https://status.cloudsync.com

### API Changelog
- **v1.3** (2024-01-15) - Added webhook support
- **v1.2** (2023-12-01) - Enhanced sync options
- **v1.1** (2023-10-15) - Added user management endpoints
- **v1.0** (2023-08-01) - Initial API release