# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

1. **DO NOT** create a public GitHub issue
2. Email security details to the maintainers
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Security Best Practices

When using Vex:

1. **API Keys**: Never hardcode API keys. Use environment variables or secure secret management.

2. **Input Validation**: Vex does not validate input text content. Ensure you sanitise user inputs before embedding.

3. **Network Security**: Embedding API calls are made over HTTPS. Ensure your network configuration doesn't downgrade connections.

4. **Rate Limiting**: Use `WithRateLimit` to prevent accidental API abuse and unexpected costs.

5. **Error Handling**: API errors may contain sensitive information. Avoid logging full error messages in production.

## Security Features

Vex is designed with security in mind:

- No credential storage (API keys passed at runtime)
- HTTPS-only API communication
- Context-aware request cancellation
- Circuit breaker protection against cascading failures
- No file system operations beyond normal Go imports

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities.
